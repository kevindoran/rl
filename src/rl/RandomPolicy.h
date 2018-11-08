#pragma once

#include "rl/Environment.h"
#include "rl/GridWorld.h"
#include "grid/Grid.h"
#include "rl/Policy.h"

namespace rl {

class RandomPolicy : public rl::Policy {
public:
    const Action& next_action(const Environment& e, const State& from_state) const override {
        return possible_actions(e, from_state).random_action();
    }

    ActionDistribution
    possible_actions(const Environment& e, const State& from_state) const override {
        ActionDistribution dist;
        for(const Action& a : e.actions()) {
            if(!e.is_action_allowed(from_state, a)) {
                continue;
            }
            dist.add_action(a);
        }
        return dist;
    }
};

} // namespace rl
