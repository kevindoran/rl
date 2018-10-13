#pragma once

#include "rl/Policy.h"
#include "rl/DeterministicPolicy.h"
#include "rl/IterativePolicyEvaluation.h"
#include "rl/StochasticPolicy.h"

namespace rl {

class PolicyIteration : public PolicyImprovement {
public:

    void set_policy_evaluation(PolicyEvaluation& evaluator) {
        evaluator = evaluator;
    }

    PolicyEvaluation& policy_evaluation() {
        return evaluator;
    }

    std::unique_ptr<Policy> improve(const Environment& env, const Policy &policy) const override {
        // We will use a StochasticPolicy object as our result.
        std::unique_ptr<StochasticPolicy> ans =
                std::make_unique<StochasticPolicy>(StochasticPolicy::create_from(env, policy));
        bool finished = false;
        while(!finished) {
            bool policy_updated = false;
            ValueFunction value_fctn = evaluator.evaluate(env, *ans);
            for(const State& s : env.states()) {
                // Skip the end states. They always have a value of 0, and we shouldn't have any
                // actions associated with it.
                if(env.is_end_state(s)) {
                    ans->clear_actions_for_state(s);
                    continue;
                }
                double v_current = value_fctn.value(s);
                for(const Action& a : env.actions()) {
                    // If the action is not possible, continue.
                    // TODO: what if you get into a dead end? Should that be allowed without it
                    // being an end state?
                    if(!env.is_action_allowed(a, s)) {
                        continue;
                    }
                    auto weights = ans->possible_actions(env, s).weight_map();
                    if(weights.size() == 1 and weights.count(&a)) {
                        continue;
                    }
                    ResponseDistribution transitions = env.transition_list(s, a);
                    double expect_value_sum = 0;
                    for(const Response& r : transitions.responses()) {
                        double value_from_next_state =
                                evaluator.discount_rate() * value_fctn.value(r.next_state);
                        expect_value_sum +=
                                r.prob_weight * (r.reward.value() + value_from_next_state);
                    }
                    Ensures(transitions.total_weight() != 0);
                    double expected_value = expect_value_sum / transitions.total_weight();
                    if(compare(expected_value, v_current, evaluator.delta_threshold()) == 1) {
                        // We found a better action!
                        // Clear all existing actions, and use the new one.
                        const int weight = 1;
                        ans->clear_actions_for_state(s);
                        ans->add_action_for_state(s, a, weight);
                        policy_updated = true;
                        // We don't override the official map for policy iteration, but we
                        // still wish to choose the best action.
                        v_current = expected_value;
                    } else {
                        if(expected_value > v_current) {
                            // Log a warning.
                            // We have found a higher value, but can't rely on it due to the
                            // value being to close to our existing value.
                        }
                    }
                }
            }
            finished = !policy_updated;
        }
        return ans;
    }


private:
    IterativePolicyEvaluation default_evalutator;
    PolicyEvaluation& evaluator = default_evalutator;
};

} // namespace rl
