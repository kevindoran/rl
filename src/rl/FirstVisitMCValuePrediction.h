#pragma once

#include "rl/Policy.h"
#include "rl/Trial.h"
#include "rl/impl/PolicyEvaluation.h"

#include <exception>
#include <limits>

namespace rl {

class FirstVisitMCValuePrediction : public impl::PolicyEvaluation {
public:
    static constexpr int MIN_VISITS = 100;


public:
    ValueFunction evaluate(const Environment& env, const Policy& p) override {
        ValueFunction ans(env.state_count());
        std::vector<int> visit_count(env.state_count(), 0);
        std::vector<double> delta(env.state_count(), std::numeric_limits<double>::max());
        // Set the value for all end states (zero).
        for(const State& end_state : env.end_states()) {
            ans.set_value(end_state, 0.0);
            delta[end_state.id()] = 0.0;
            visit_count[end_state.id()] = std::numeric_limits<int>::max();
        }
        // This algorithm will use exploring starts (start states) in order to ensure we get
        // value estimates for all states even if our policy is deterministic.
        double max_delta = std::numeric_limits<double>::max();
        int min_visit = 0;
        while(max_delta > delta_threshold_ or min_visit < MIN_VISITS) {
            for(const State& start_state : env.states())
            {
                if(env.is_end_state(start_state))   {
                    continue;
                }
                Trace trace = run_trial(env, p, start_state);
                update_value_fctn(ans, visit_count, delta, trace);
                max_delta = *std::max_element(delta.begin(), delta.end());
                min_visit = *std::min_element(visit_count.begin(), visit_count.end());
            }
        }
        return ans;
    }

    void set_discount_rate(double discount_rate) override {
        throw std::runtime_error("This evaluator only supports episodic tasts.");
    }

    double discount_rate() const override {
        return 1.0;
    }

    static void update_value_fctn(
            ValueFunction& value_fctn,
            std::vector<int>& visit_count,
            std::vector<double>& delta,
            const Trace& trace) {
        double retrn = 0;
        Expects(!trace.empty());
        // Track the first occurrence of a state so that we can implement first-visit (skip states
        // that have been visited already).
        std::unordered_map<ID, int> first_occurrence;
        for(std::size_t i = 0; i < trace.size(); i++) {
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
            if(first_occurrence[step.state.id()] < static_cast<int>(i)) {
                // We still need to maintain the correct return value.
                retrn += step.reward;
                continue;
            }
            double current_value = value_fctn.value(step.state);
            double n = ++visit_count[step.state.id()];
            double updated_value = current_value + 1/n * (retrn - current_value);
            value_fctn.set_value(step.state, updated_value);
            delta[step.state.id()] = error_as_factor(current_value, updated_value);
            retrn += step.reward;
        }
    }
};

} // namespace rl
