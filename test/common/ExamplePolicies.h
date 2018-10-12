#pragma once

#include "rl/Policy.h"

namespace rl {
namespace test {

/**
 * A policy that always choose the first action in an environment.
 */
class FirstActionPolicy : public rl::Policy {
public:
    const Action& next_action(const Environment& e, const State& from_state) const override {
        Expects(e.action_count());
        const Action& action = *e.actions_begin();
        return action;
    }

    ActionDistribution
    possible_actions(const Environment& e, const State& from_state) const override {
        return ActionDistribution::single_action(next_action(e, from_state));
    }
};

/**
 * A faulty policy that returns an empty ActionDistribution.
 */
class NoActionPolicy : public rl::Policy {
public:
    const Action& next_action(const Environment& e, const State& from_state) const override {
        Expects(e.action_count());
        return *e.actions_begin();
    }

    ActionDistribution
    possible_actions(const Environment& e, const State& from_state) const override {
        return ActionDistribution();
    }
};

/**
 * A faulty policy that returns an action Distribution with an action having zero weight.
 */
class ZeroWeightActionPolicy : public rl::Policy {
    const Action& next_action(const Environment& e, const State& from_state) const override {
        Expects(e.action_count());
        return *e.actions_begin();
    }

    ActionDistribution
    possible_actions(const Environment& e, const State& from_state) const override {
        ActionDistribution ans;
        long weight = 0;
        ans.add_action(next_action(e, from_state), weight);
        return ans;
    }
};

} // namespace test
} // namespace rl
