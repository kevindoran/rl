#pragma once

#include "Policy.h"
#include "StochasticPolicy.h"
#include "FirstVisitMCActionValuePredictor.h"

namespace rl {

/**
 * A basic policy improver which uses an state-action value function.
 *
 * It is interesting to compare the implementation of the below policy improver with an improver
 * that works with state value functions (e.g. PolicyIterator). The latter implementation is far
 * more involved and requires the environment to have fully specified dynamics.
 */
class ActionValuePolicyIterator : public PolicyImprovement {
public:

    std::unique_ptr<Policy> improve(const Environment& env, const Policy& policy) const override {
        std::unique_ptr<StochasticPolicy> ans =
                std::make_unique<StochasticPolicy>(StochasticPolicy::create_from(env, policy));
        bool finished = false;
        while(!finished) {
            bool policy_updated = false;
            const ActionValueFunction& value_fctn = evaluate(default_evaluator, env, policy);
            for(const State& state : env.states()) {
                // Skip end states, for no action can be taken from them.
                if(env.is_end_state(state)) {
                    continue;
                }
                const Action* best_action = nullptr;
                double best_value = std::numeric_limits<double>::min();
                for(const Action& action : env.actions()) {
                    if(!env.is_action_allowed(state, action)) {
                        // TODO: what is to be done in states that have no valid actions?
                        continue;
                    }
                    double v = value_fctn.value(state, action);
                    if(greater_than(v, best_value, evaluator_.delta_threshold())) {
                        p_best_action = &action;
                        best_value = v;
                    }
                }
                if(best_action) {
                    policy_updated = true;
                }
            }
            finished = !policy_updated && evaluator_.finished();
        }
        return ans;
    }

    const ActionBasedEvaluator& policy_evaluator() const override {
        return evaluator_;
    }

    ActionBasedEvaluator& policy_evaluator() override {
        return const_cast<ActionBasedEvaluator&>(
                static_cast<const ActionValuePolicyIterator*>(this)->policy_evaluator());
    }

private:
    FirstVisitMCActionValuePredictor default_evaluator;
    ActionBasedEvaluator& evaluator_ = default_evaluator;
};

}

