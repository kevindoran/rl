#pragma once

#include "rl/Policy.h"
#include "rl/Trial.h"
#include "rl/impl/PolicyEvaluator.h"

#include <exception>
#include <limits>

namespace rl {

class FirstVisitMCValuePredictor : public StateBasedEvaluator,
                                    public impl::PolicyEvaluator {
public:
    static constexpr int MIN_VISITS = 100;

public:
    void initialize(const Environment& env, const Policy& policy) override {
        impl::PolicyEvaluator::initialize(env, policy);
        value_fuction_ = ValueFunction(env.state_count());
        visit_count = std::vector<int>(env.state_count(), 0);
        delta = std::vector<double>(env.state_count(), std::numeric_limits<double>::max());
        // Set the value for all end states (zero).
        for(const State& end_state : env.end_states()) {
            delta[end_state.id()] = 0.0;
            visit_count[end_state.id()] = std::numeric_limits<int>::max();
        }
    }

    void step() override {
        const Environment& env = *CHECK_NOTNULL(env_);
        const Policy& policy = *CHECK_NOTNULL(policy_);
        // This algorithm will use exploring starts (start states) in order to ensure we get
        // value estimates for all states even if our policy is deterministic.
        for(const State& start_state : env.states())
        {
            if(env.is_end_state(start_state))   {
                continue;
            }
            Trace trace = run_trial(env, policy, &start_state);
            update_value_fctn(trace);
        }
        most_recent_delta_ = *std::max_element(delta.begin(), delta.end());
        min_visit_ = *std::min_element(visit_count.begin(), visit_count.end());
        steps_++;
    }

    void run() override {
        while(most_recent_delta_ > delta_threshold_ or min_visit_ < MIN_VISITS) {
            step();
        }
    }

    const ValueFunction& value_function() const override {
        return value_fuction_;
    }

    void set_discount_rate(double discount_rate) override {
        throw std::runtime_error("This evaluator only supports episodic tasts.");
    }

    double discount_rate() const override {
        return 1.0;
    }

private:
    void update_value_fctn(const Trace& trace) {
        double retrn = 0;
        Expects(!trace.empty());
        // Track the first occurrence of a state so that we can implement first-visit (skip states
        // that have been visited already).
        std::unordered_map<ID, int> first_occurrence;
        // We can skip the last state, as you can't leave an end state.
        for(std::size_t i = 0; i < trace.size()-1; i++) {
            const State& s = trace[i].state;
            if(!first_occurrence.count(s.id())) {
                Ensures(i <= std::numeric_limits<int>::max());
                first_occurrence[s.id()] = static_cast<int>(i);
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
            if(first_occurrence[step.state.id()] < i) {
                // We still need to maintain the correct return value.
                retrn += step.reward;
                continue;
            }
            double current_value = value_fuction_.value(step.state);
            double n = ++visit_count[step.state.id()];
            Ensures(n > 0);
            double updated_value = current_value + 1/n * (retrn - current_value);
            value_fuction_.set_value(step.state, updated_value);
            delta[step.state.id()] = std::abs(current_value - updated_value);
            retrn += step.reward;
        }
    }

private:
    ValueFunction value_fuction_;
    std::vector<int> visit_count{};
    std::vector<double> delta{};
    long min_visit_ = 0;
};

} // namespace rl
