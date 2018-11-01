#pragma once

#include "rl/Environment.h"
#include "rl/GridWorld.h"
#include "grid/Grid.h"
#include "rl/Policy.h"

namespace rl {

/**
 * A policy for a GridWorld which chooses a random action uniformly from (up, down, left, right).
 *
 * Improvements:
 *    - If GridWorld implemented an interface, we wouldn't need to use a class template.
 */
// TODO: is this class needed? We have RandomPolicy?
template<int H, int W>
class RandomGridPolicy : public Policy {
public:
    explicit RandomGridPolicy(const GridWorld<H, W> &grid_world) : grid_world_(grid_world) {}

    const Action & next_action(const Environment &e, const State &from_state) const override {
        return possible_actions(e, from_state).random_action();
    }

    ActionDistribution
    possible_actions(const Environment &e, const State &from_state) const override {
        ActionDistribution dist;
        grid::Position from = grid_world_.state_to_pos(from_state);
        for (grid::Direction dir : grid::directions) {
            if (!grid_world_.is_movement_valid(from, dir)) {
                continue;
            }
            dist.add_action(grid_world_.dir_to_action(dir));
        }
        return dist;
    }

private:
    const GridWorld<H, W> &grid_world_;
};

} // namespace rl
