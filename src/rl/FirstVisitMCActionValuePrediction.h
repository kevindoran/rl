#pragma once

#include "rl/ActionValueFunction.h"
#include "rl/Policy.h"
#include "rl/Trial.h"

namespace rl {

// Long enough name?
class FirstVisitMCActionValuePrediction : ActionValuePolicyEvaluation {

public:
    static constexpr double DEFAULT_DELTA_THRESHOLD = 0.00001;
    static constexpr double DEFAULT_DISCOUNT_RATE = 1.0;
    static constexpr int MIN_VISIT = 100;

    ActionValueFunction evaluate(const Environment& env, const Policy& policy) override {
        // We will use first-visit & exploring starts.
        ActionValueFunction ans(env.state_count(), env.action_count());
        std::vector<int> visit_count(env.state_count() * env.action_count(), 0);
        std::vector<double> delta(env.state_count() * env.action_count(), 0);
        // Set the value for all end states (zero).
        for(const State& state : env.end_states()) {
            for(const Action& action : env.actions()) {
                delta[hash(state, action, env.action_count())] = 0.0;
                visit_count[hash(state, action, env.action_count())] =
                        std::numeric_limits<int>::max();
            }
        }
        int min_visit = 0;
        double max_delta = std::numeric_limits<double>::max();
        while(max_delta > delta_threshold_ or min_visit < MIN_VISIT) {
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
                    update_action_value_fctn(ans, visit_count, delta, trace, env.action_count());
                }
            }
            // Update stopping criteria.
            max_delta = *std::max_element(delta.begin(), delta.end());
            min_visit = *std::min_element(visit_count.begin(), visit_count.end());
        }
        return ans;
    }

    void set_discount_rate(double discount_rate) override {
        throw std::runtime_error("This evaluator only supports episodic tasts.");
    }

    double discount_rate() const override {
        return 1.0;
    }

    void set_delta_threshold(double delta_threshold) override {
        delta_threshold_ = delta_threshold;
    }

    double delta_threshold() const override {
        return delta_threshold_;
    }

private:
    void update_action_value_fctn(
            ActionValueFunction& value_fctn,
            std::vector<int>& visit_count,
            std::vector<double>& delta,
            const Trace& trace,
            ID action_count) {
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
            if(!first_occurrence.count(hash(state, action, action_count))) {
                Ensures(i <= std::numeric_limits<int>::max());
                first_occurrence[hash(state, action, action_count)] = static_cast<int>(i);
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
            long hash_val = hash(state, action, action_count);
            if(first_occurrence[hash_val] < i) {
                // We still need to maintain the correct return value.
                retrn += step.reward;
                continue;
            }
            double current_value = value_fctn.value(state, action);
            double n = ++visit_count[hash_val];
            Ensures(n > 0);
            double updated_value = current_value + 1/n * (retrn - current_value);
            value_fctn.set_value(state, action, updated_value);
            delta[hash_val] = error_as_factor(current_value, updated_value);
            retrn += step.reward;
        }
    }

    static long hash(const State& state, const Action& action, ID action_count) {
        return state.id() * action_count + action.id();
    }

protected:
    double delta_threshold_ = DEFAULT_DELTA_THRESHOLD;
    double discount_rate_ = DEFAULT_DISCOUNT_RATE;

};

} // namespace rl
