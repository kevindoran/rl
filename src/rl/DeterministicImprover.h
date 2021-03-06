#pragma once

#include "rl/Policy.h"
#include "rl/DeterministicPolicy.h"
#include "rl/IterativePolicyEvaluator.h"
#include "rl/StochasticPolicy.h"

namespace rl {

class DeterministicImprover : public PolicyImprover {
public:

    void set_discount_rate(double discount_rate) override {
        evaluator_.set_discount_rate(discount_rate);
    }

    double discount_rate() const override {
        return evaluator_.discount_rate();
    }

    void set_delta_threshold(double max_delta) override {
        evaluator_.set_delta_threshold(max_delta);
    }

    double delta_threshold() const override {
        return evaluator_.delta_threshold();
    }

    void set_policy_evaluator(StateBasedEvaluator& evaluator) {
        evaluator_ = evaluator;
    }

    const PolicyEvaluator& policy_evaluator() const {
        return evaluator_;
    }

    PolicyEvaluator& policy_evaluator() {
        return const_cast<PolicyEvaluator&>(
                static_cast<const DeterministicImprover*>(this)->policy_evaluator());
    }

    std::unique_ptr<Policy> improve(const Environment& env, const Policy &policy) const override {
        // note: It is interesting to see how I approached this on the first try. The input policy
        // is copied, and that becomes the starting point to iterate from. The copied input policy
        // is stored as a StochasticPolicy and this object becomes the returned result.
        // In comparison, I've seen other implementations where:
        //    * on the first iteration, the input policy is used to decide actions.
        //    * on all subsequent iterations, a greedy policy wrt the value function is used.
        // With this approach, no storage space is used for the policy until returning a result.
        // Instead, the current best state_value_function is stored.
        // In addition, there is no need to create a stochastic policy from the input policy.

        // We will use a StochasticPolicy object as our result.
        std::unique_ptr<StochasticPolicy> ans =
                std::make_unique<StochasticPolicy>(StochasticPolicy::create_from(env, policy));
        bool finished = false;
        while(!finished) {
            bool policy_updated = false;
            const ValueTable& value_fctn = evaluate(evaluator_, env, *ans);
            for(const State& s : env.states()) {
                // Skip the end states. They always have a value of 0, and we shouldn't have any
                // actions associated with it.
                if(env.is_end_state(s)) {
                    ans->clear_actions_for_state(s);
                    continue;
                }
                // Check if the current policy is deterministic in this state (used by
                // calculate_best_action).
                const Action* current_action = nullptr;
                if(ans->possible_actions(env, s).action_count() == 1) {
                    current_action = &ans->next_action(env, s);
                }
                const Action* improved_action = nullptr;
                double reward = 0;
                std::tie(improved_action, reward) = calculate_best_action(
                        env, s, value_fctn, current_action);
                if(improved_action) {
                    // We found a better action!
                    // Clear all existing actions, and use the new one.
                    const Weight weight = 1;
                    ans->clear_actions_for_state(s);
                    ans->add_action_for_state(s, *CHECK_NOTNULL(improved_action), weight);
                    policy_updated = true;
                }
            }
            finished = !policy_updated;
        }
        return ans;
    }

private:
    const std::pair<const Action*, double> calculate_best_action(
            const Environment& env,
            const State& from_state,
            const ValueTable& value_fctn,
            const Action* current_action) const {
        std::pair<const Action*, double> ans{nullptr, 0};
        for(const Action& a : env.actions()) {
            // If the action is not possible, continue.
            // TODO: what if you get into a dead end? Should that be allowed without it being an end
            // state?
            if(!env.is_action_allowed(from_state, a)) {
                continue;
            }
            // We already know the value for this action: v_current.
            if(current_action and *current_action == a) {
                continue;
            }
            double expected_value = calculate_reward(env, from_state, a, value_fctn);
            double v_current = value_fctn.value(from_state);
            if(greater_than(expected_value, v_current, evaluator_.delta_threshold())) {
                // We found a better action!
                ans = {&a, expected_value};
            } else {
                LOG_IF(WARNING, expected_value > v_current)
                    << "We have found a higher value but can't rely on it due to the value being "
                       "too close to our existing value. The difference is less than the evaluator "
                       "threshold";
            }
        }
        return ans;
    }

    double calculate_reward(const Environment& env, const State& from_state,
            const Action& action, const ValueTable& value_fctn) const {
        ResponseDistribution transitions = env.transition_list(from_state, action);
        double expect_value_sum = 0;
        for(const Response& r : transitions.responses()) {
            double next_state_value = evaluator_.discount_rate() * value_fctn.value(r.next_state);
            expect_value_sum += r.prob_weight * (r.reward.value() + next_state_value);
        }
        Ensures(transitions.total_weight() != 0);
        double expected_value = expect_value_sum / transitions.total_weight();
        return expected_value;
    }

private:
    IterativePolicyEvaluator default_evalutator;
    StateBasedEvaluator& evaluator_ = default_evalutator;
};

} // namespace rl
