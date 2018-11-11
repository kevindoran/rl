#pragma once

#include "rl/Policy.h"

namespace rl {

class BlendedPolicy : public Policy {
public:
    // note: this can be extended from 2 policies to n policies.
    // note: taking the policies in by const pointer (rather than reference) so that it is clear
    // that we expect the client to maintain the lifetime of these objects. a const ref type
    // would allow the client to use an rvalue temporary as an argument, which would lead to
    // undefined behaviour.
    BlendedPolicy(const Policy* policy1, const Policy* policy2, double blend) :
    policy1(*CHECK_NOTNULL(policy1)), policy2(*CHECK_NOTNULL(policy2)), blend(blend)
    {
        CHECK_GE(1.0, blend);
        CHECK_LE(0.0, blend);
    }

    const Action& next_action(const Environment& env, const State& from_state) const override {
        return possible_actions(env, from_state).random_action();
    }

    ActionDistribution
    possible_actions(const Environment& env, const State& from_state) const override {
        // Add policy1.
        ActionDistribution res;
        for(const Action& a : env.actions()) {
            Weight weight1 = policy1.possible_actions(env, from_state).probability(a);
            Weight weight2 = policy2.possible_actions(env, from_state).probability(a);
            Weight new_weight = (1.0-blend) * weight1 + blend * weight2;
            if(new_weight != 0.0) {
                res.add_action(a, new_weight);
            }
        }
        return res;
    }

private:
    // Switch to const* if assignment operator or move ctr is needed.
    const Policy& policy1;
    const Policy& policy2;
    double blend;
};

} // namespace rl