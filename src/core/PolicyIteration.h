#ifndef REINFORCEMENT_POLICYITERATION_H
#define REINFORCEMENT_POLICYITERATION_H

#include "core/Policy.h"
#include "core/DeterministicPolicy.h"
#include "core/IterativePolicyEvaluation.h"
#include "core/StochasticPolicy.h"

namespace rl {

class PolicyIteration : public PolicyImprovement {
public:

    void set_policy_evaluation(PolicyEvaluation& evaluator) {
        evaluator = evaluator;
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
                for(const Action& a : env.actions()) {
                    TransitionDistribution transitions = env.transition_list(s, a);
                    // If the action is not possible, clear actions and continue.
                    if(!transitions.total_weight) {
                        ans->clear_actions_for_state(s);
                        continue;
                    }
                    double expect_value_sum = 0;
                    for(const Transition& t : transitions.transitions) {
                        double value_from_next_state =
                                evaluator.discount_rate() * value_fctn.value(t.next_state());
                        expect_value_sum +=
                                t.prob_weight() * (t.reward().value() + value_from_next_state);
                    }
                    double expected_value = expect_value_sum / transitions.total_weight;
                    double v_current = value_fctn.value(s);
                    if(expected_value > v_current) {
                        // We found a better action!
                        // Clear all existing actions, and use the new one.
                        const int weight = 1;
                        ans->clear_actions_for_state(s);
                        ans->add_action_for_state(s, a, weight);
                        policy_updated = true;
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

#endif //REINFORCEMENT_POLICYITERATION_H
