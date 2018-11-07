#pragma once

#include "rl/Policy.h"
#include "rl/DeterministicPolicy.h"
#include "rl/GridWorld.h"

namespace rl {
namespace test {

/**
 * A policy that always choose the first action in an environment.
 */
 // TODO: Is this needed if we have FirstValidActionPolicy?
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

class FirstValidActionPolicy : public rl::Policy {
public:
    const rl::Action&
    next_action(const rl::Environment& e, const rl::State& from_state) const override {
        const Action* res;
        for(const rl::Action& a : e.actions()) {
            if(!e.is_action_allowed(from_state, a)) {
                continue;
            }
            res = &a;
        }
        return *CHECK_NOTNULL(res);
    }

    ActionDistribution
    possible_actions(const rl::Environment &e, const rl::State &from_state) const override {
        return ActionDistribution::single_action(next_action(e, from_state));
    }
};

class ConstantPolicy : public rl::Policy {
public:
    explicit ConstantPolicy(ID action_id) : action_id(action_id)
    {}

    const Action& next_action(const Environment& e, const State& from_state) const override {
        CHECK_GT(e.action_count(), action_id);
        const Action& action = e.action(action_id);
        CHECK(e.is_action_allowed(from_state, action));
        return action;
    }

    ActionDistribution
    possible_actions(const Environment& e, const State& from_state) const override {
        return ActionDistribution::single_action(next_action(e, from_state));
    }

private:
    ID action_id;
};

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
public:
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

// note: We can make GridWord inherit from an abstract class allowing methods to use the interface
// instead of having to become templated to deal with the W & H params.
template<int W, int H>
rl::DeterministicLambdaPolicy create_down_up_policy(const rl::GridWorld<W,H>& grid_world) {
    // note: if the return type is not specified, the action gets returned by value, which
    // leads to an error later on when the reference is used.
    auto fctn = [&grid_world](const rl::Environment& e, const rl::State& s) -> const rl::Action& {
        grid::Position pos = grid_world.state_to_pos(s);
        // We can't just go down, as we will get an exception trying to go outside the grid.
        bool can_go_down = grid_world.grid().is_valid(pos.adj(grid::Direction::DOWN));
        if (can_go_down) {
            return grid_world.dir_to_action(grid::Direction::DOWN);
        } else {
            return grid_world.dir_to_action(grid::Direction::UP);
        }
    };
    return rl::DeterministicLambdaPolicy(fctn);
}

} // namespace test
} // namespace rl
