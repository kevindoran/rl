#pragma once

#include <rl/impl/PolicyEvaluator.h>
#include "rl/ActionValueFunction.h"
#include "rl/Policy.h"
#include "rl/Trial.h"

namespace rl {

// Long enough name?
class FirstVisitMCActionValuePredictor : public ActionBasedEvaluator,
                                          public impl::PolicyEvaluator {

public:
    static constexpr double DEFAULT_DELTA_THRESHOLD = 0.00001;
    static constexpr double DEFAULT_DISCOUNT_RATE = 1.0;
    static constexpr int MIN_VISIT = 100;

public:

    void initialize(const Environment& env, const Policy& policy) override {
        impl::PolicyEvaluator::initialize(env, policy);
        // We use action_count * state_count elements of the arrays.
        // Check that (action_count * state_count) < array size max.
        // TODO: how to best guard against overflow here?
        // Expects(std::numeric_limits<std::size_t>::max() / env.action_count() > env.state_count());
        Ensures(env.action_count() * env.state_count() > 0);
        std::size_t element_count = env.action_count() * env.state_count();
        value_function_ = ActionValueFunction(env.state_count(), env.action_count());
        visit_count = std::vector<int>(element_count, 0);
        delta = std::vector<double>(element_count, 0);
        // Set the value for all end states (zero).
        for(const State& state : env.end_states()) {
            for(const Action& action : env.actions()) {
                delta[hash(state, action)] = 0.0;
                visit_count[hash(state, action)] =
                        std::numeric_limits<int>::max();
            }
        }
    }

    void step() override {
        const Environment& env = *CHECK_NOTNULL(env_);
        const Policy& policy = *CHECK_NOTNULL(policy_);
        // We will use first-visit & exploring starts.
        // Force starting from all state-action pairs.
        for(const State& start_state : env.states()) {
            // Skip end states, as no action can be taken from them.
            if(env.is_end_state(start_state)) {
                continue;
            }
            for(const Action& start_action : env.actions()) {
                // Skip state-action pairs that are not valid for the environment.
                if(!env.is_action_allowed(start_state, start_action)) {
                    continue;
                }
                Trace trace = run_trial(env, policy, &start_state, &start_action);
                update_action_value_fctn(trace);
            }
        }
        // Update stopping criteria.
        most_recent_delta_ = *std::max_element(delta.begin(), delta.end());
        min_visit_ = *std::min_element(visit_count.begin(), visit_count.end());
    }

    void run() override {
        while(most_recent_delta_ > delta_threshold_ or min_visit_ < MIN_VISIT) {
            step();
        }
    }

    const ActionValueFunction& value_function() const override {
        return value_function_;
    }

    void set_discount_rate(double discount_rate) override {
        throw std::runtime_error("This evaluator only supports episodic tasts.");
    }

    double discount_rate() const override {
        return 1.0;
    }

private:
    void update_action_value_fctn(const Trace& trace) {
        double retrn = 0;
        Expects(!trace.empty());
        // Track the first occurrence of a state so that we can implement first-visit (skip states
        // that have been visited already).
        // This map assumes that max(long) > (state_count * action_count).
        Ensures(std::numeric_limits<long>::max() >= visit_count.size());
        std::unordered_map<long, int> first_occurrence;
        // We can skip the last state (end state). There is no exit action paired with an end state.
        for(std::size_t i = 0; i < trace.size() - 1; i++) {
            const State& state = trace[i].state;
            const Action& action = *CHECK_NOTNULL(trace[i].action);
            if(!first_occurrence.count(hash(state, action))) {
                Ensures(i <= std::numeric_limits<int>::max());
                first_occurrence[hash(state, action)] = static_cast<int>(i);
            }
        }
        // Add the reward for entering the end state.
        retrn += trace.back().reward;
        // Iterate backwards over the time steps, starting from one before the end.
        // Don't use size_t here, as you will have an infinite loop given that it's unsigned.
        Ensures(trace.size() <= std::numeric_limits<int>::max());
        for(int i = static_cast<int>(trace.size() - 2); i >= 0; i--) {
            TimeStep step = trace[i];
            // First visit check. Skip this step if the state occurs in an earlier step.
            // Without this check, we would be implementing every-visit.
            const State& state = step.state;
            const Action& action = *CHECK_NOTNULL(step.action);
            long hash_val = hash(state, action);
            if(first_occurrence[hash_val] < i) {
                // We still need to maintain the correct return value.
                retrn += step.reward;
                continue;
            }
            double current_value = value_function_.value(state, action);
            double n = ++visit_count[hash_val];
            Ensures(n > 0);
            double updated_value = current_value + 1/n * (retrn - current_value);
            value_function_.set_value(state, action, updated_value);
            delta[hash_val] = error_as_factor(current_value, updated_value);
            retrn += step.reward;
        }
    }

    long hash(const State& state, const Action& action) {
        return state.id() * CHECK_NOTNULL(env_)->action_count() + action.id();
    }

private:
    ActionValueFunction value_function_;
    std::vector<int> visit_count{};
    std::vector<double> delta{};
    long min_visit_ = 0;

};

} // namespace rl
