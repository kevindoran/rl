#pragma once

#include <limits>

#include "rl/Policy.h"
#include "rl/impl/PolicyEvaluator.h"

namespace rl {

class IterativePolicyEvaluator : public StateBasedEvaluator,
                                 public impl::PolicyEvaluator {
public:

    void initialize(const Environment& env, const Policy& policy) override {
        impl::PolicyEvaluator::initialize(env, policy);
        value_function_ = ValueFunction(env.state_count());
    }

    /**
     * Run a single step of the algorithm.
     *
     * Pseudo-code:
     * error = inf
     * while(error > threshold)
     *    error = 0
     *    for(s in states)
     *        val = 0
     *        actions = policy.get_actions(from_state)
     *        for(a in actions)
     *            transitions = env.get_possible_transitions(from_state, a)
     *            for(t in transitions)
     *                val += (t.reward + value[t.next_state]) * t.prob)
     *        error = max(error, |old_s - val|)
     *        values[s] = val
     */
    void step() override {
        // Check the env_ & policy_ pointers once, then give them a shorthand.
        const Environment& e = *CHECK_NOTNULL(env_);
        const Policy& p = *CHECK_NOTNULL(policy_);
        double error = 0;
        for(const State& s : e.states()) {
            if(e.is_end_state(s)) {
                continue;
            }
            double expected_value = 0;
            Policy::ActionDistribution action_dist = p.possible_actions(e, s);
            // A policy must have an action for every non-end state.
            Expects(action_dist.action_count());
            // The action_dist can't have zero weight in total.
            Expects(action_dist.total_weight());
            for(auto action_weight_pair : action_dist.weight_map()) {
                const Action& action = *CHECK_NOTNULL(action_weight_pair.first);
                long action_weight = action_weight_pair.second;
                // A policy's actions can't have zero weight.
                Expects(action_weight);
                ResponseDistribution response_dist = e.transition_list(s, action);
                for(const Response& r : response_dist.responses()) {
                    double transition_reward = r.reward.value();
                    double reward_from_target = value_function_.value(r.next_state);
                    // Looking at Compiler Explorer for the following computation it seems that
                    // compilers will not optimize (a / b) * (c / d) into (a * c) / (b * d).
                    // It looks like it can't as there will be differences due to rounding etc.
                    // I'll do it manually instead.
                    // (link to results: https://godbolt.org/z/v8WiUN )
                    /*
                    double probability =
                            (action_weight / (double) action_dist.total_weight) *
                            (t.get().prob_weight() / (double) response_dist.total_weight);
                    */
                    double denominator =
                            (action_dist.total_weight() * (double) response_dist.total_weight());
                    Ensures(denominator != 0);
                    double probability = (action_weight * (double) r.prob_weight) / denominator;
                    expected_value += transition_reward * probability;
                    expected_value += reward_from_target * discount_rate_ * probability;
                }
            }
            double prev = value_function_.value(s);
            error = std::max(error, error_as_factor(prev, expected_value));
            value_function_.set_value(s, expected_value);
        }
        most_recent_delta_ = error;
    }

    const ValueFunction& value_function() const override {
        return value_function_;
    }

private:
    ValueFunction value_function_;
};

} // namespace rl
