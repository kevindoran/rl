#pragma once

#include <glog/logging.h>

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
class ActionValuePolicyImprover : public PolicyImprover {
public:
    std::unique_ptr<Policy> improve(const Environment& env, const Policy& policy) const override {
        std::unique_ptr<StochasticPolicy> ans =
                std::make_unique<StochasticPolicy>(StochasticPolicy::create_from(env, policy));
        bool finished = false;
        evaluator_.initialize(env, *ans);
        int loop = 0;
        while(!finished) {
            bool policy_updated = false;
            //const ActionValueFunction& value_fctn = evaluate(evaluator_, env, *ans);
            // TODO: Without loop detection, this call risks a possible infinite loop. For example,
            // in a deterministic grid world, a trial might go:
            //     (0,0), (0,1), (0,2), (0,3), (1, 3), (2, 3), (3, 3 [end]).
            // This trial could make 'right' the highest scoring action for (0,2). Another trial
            // might
            // 0, while another trial might pass by tile
            evaluator_.step();
            const ActionValueFunction& value_fctn = evaluator_.value_function();
            for(const State& state : env.states()) {
                // Skip end states, for no action can be taken from them.
                if(env.is_end_state(state)) {
                    ans->clear_actions_for_state(state);
                    continue;
                }
                // At the moment, the algorithm will wipe any multi-action policies- it doesn't
                // bother to check if all actions have the same value and therefore might be
                // all optimal.
                // TODO: support maintaining multi-action policies.
                double best_value;
                if(ans->possible_actions(env, state).action_count() > 1) {
                    best_value = std::numeric_limits<double>::lowest();
                }
                else {
                    const Action& current_action = ans->possible_actions(env, state).any();
                    best_value = value_fctn.value(state, current_action);
                }
                const Action* p_best_action = nullptr;
                int allowed_actions = 0;
                for(const Action& action : env.actions()) {
                    if(!env.is_action_allowed(state, action)) {
                        continue;
                    }
                    allowed_actions++;
                    double v = value_fctn.value(state, action);
                    if(greater_than(v, best_value, evaluator_.delta_threshold())) {
                        p_best_action = &action;
                        best_value = v;
                    }
                }
                LOG_IF(ERROR, !allowed_actions) << "A state was encountered from which there were"
                    "no allowed actions to be taken. State: " << state.name();
                if(p_best_action) {
                    // We found a better action.
                    ans->clear_actions_for_state(state);
                    const Weight weight = 1;
                    ans->add_action_for_state(state, *CHECK_NOTNULL(p_best_action), weight);
                    policy_updated = true;
                }
            }
            loop++;
            finished = !policy_updated && evaluator_.finished();
        }
        return ans;
    }

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

    const ActionBasedEvaluator& policy_evaluator() const {
        return evaluator_;
    }

    ActionBasedEvaluator& policy_evaluator() {
        return const_cast<ActionBasedEvaluator&>(
                static_cast<const ActionValuePolicyImprover*>(this)->policy_evaluator());
    }

    void set_policy_evaluator(const ActionBasedEvaluator& evaluator) {
        evaluator_ = evaluator;
    }

private:
    FirstVisitMCActionValuePredictor default_evaluator;
    ActionBasedEvaluator& evaluator_ = default_evaluator;
};

} // namespace rl

