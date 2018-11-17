#pragma once

#include "rl/GridWorld.h"
#include "rl/Trial.h"

namespace rl {
namespace test {
namespace suttonbarto {

const int CLIFF_WORLD_HEIGHT = 4;
const int CLIFF_WORLD_WIDTH = 12;

/**
 * The CliffWorld environment is used to highlight a difference in the behaviour of
 * Sarsa (on-policy, using e-greedy action selection) and Q-learning (off-policy) algorithms:
 * Q-Learning will find the optimal route even though the route is alone by the cliff (which has a
 * very high negative reward). Sarsa with e-greedy will take a safer but not optimal route.
 * This safer route is taken as Sarsa will under-rate the optimal route. This is due to the random
 * action sometimes causing the route off the cliff to be taken. Q-Learning, being an off-policy
 * algorithm, doesn't have this characteristic. While Q-learning will find the optimal policy,
 * Sarsa will have better online performance.
 *
 * Refer to p132 of Sutton & Barto 2018.
 */
class CliffWorld : public GridWorld<CLIFF_WORLD_HEIGHT, CLIFF_WORLD_WIDTH> {
public:
    static constexpr int HEIGHT = 4;
    static constexpr int WIDTH = 12;
    using GridType = GridWorld<HEIGHT, WIDTH>::GridType;
    static constexpr double TRANSITION_REWARD = -1;
    static constexpr double FALL_REWARD = -100;
    static constexpr grid::Position GOAL_POS{3,11};
    static constexpr grid::Position START_POS{3,0};
    static constexpr int CLIFF_ROW = 3;
public:
    CliffWorld() {
        set_all_rewards_to(TRANSITION_REWARD);
        mark_as_end_state(pos_to_state(GOAL_POS));
        set_start_state(pos_to_state(START_POS));
    }

    Response next_state(const State& from_state, const Action& action) const override {
        Response standard_response = GridWorld::next_state(from_state, action);
        if(is_cliff_tile(standard_response.next_state)) {
            // TODO: refactor when implementing non-stored rewards.
            const Reward reward_local(-1, FALL_REWARD);
            Response cliff_response{start_state(), reward_local, 1.0};
            return cliff_response;
        } else {
            return standard_response;
        }
    }

private:
    bool is_cliff_tile(const State& state) const {
        if(is_end_state(state) or state == start_state()) {
            return false;
        }
        grid::Position pos = state_to_pos(state);
        bool is_cliff = pos.y == CLIFF_ROW;
        return is_cliff;
    }
};

} // namespace suttonbarto
} // namespace test
} // namespace rl
