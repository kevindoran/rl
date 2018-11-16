#pragma once

#include "rl/GridWorld.h"
#include "rl/Trial.h"
#include "TestEnvironment.h"

namespace rl {
namespace test {
namespace suttonbarto {

/**
 * Example 6.6: Cliff Walking.
 *
 * Refer to p132 of Sutton & Barto 2018.
 *
 * The CliffWorld environment is used to highlight a difference in the behaviour of
 * Sarsa (on-policy, using e-greedy action selection) and Q-learning (off-policy) algorithms:
 * Q-Learning will find the optimal route even though the route is alone by the cliff (which has a
 * very high negative reward). Sarsa with e-greedy will take a safer but not optimal route.
 * This safer route is taken as Sarsa will under-rate the optimal route. This is due to the random
 * action sometimes causing the route off the cliff to be taken. Q-Learning, being an off-policy
 * algorithm, doesn't have this characteristic. While Q-learning will find the optimal policy,
 * Sarsa will have better online performance.
 */
class Example6_6 : public TestEnvironment {
public:
    static constexpr int HEIGHT = 4;
    static constexpr int WIDTH = 12;
    using GridType = GridWorld<HEIGHT, WIDTH>::GridType;
    static constexpr double TRANSITION_REWARD = -1;
    static constexpr double FALL_REWARD = -100;
    static constexpr grid::Position GOAL_POS{3,11};
    static constexpr grid::Position START_POS{3,0};
    static constexpr int CLIFF_ROW = 3;
    static const std::vector<grid::Position> SAFE_ROUTE;
    static const std::vector<grid::Position> OPTIMAL_ROUTE;

    class CliffWorld : public GridWorld<HEIGHT, WIDTH> {
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

    std::string name() const override {
        return "Example 6.6: Cliff Walking";
    }

    const CliffWorld& env() const override {
        return env_;
    }

    double required_discount_rate() const override {
        return 1.0;
    }

    double required_delta_threshold() const override {
        return 0.001;
    }

    OptimalActions optimal_actions(const State& from_state) const override {
        OptimalActions ans;
        std::transform(
                std::begin(optimal_actions_[from_state.id()]),
                std::end(optimal_actions_[from_state.id()]),
                std::inserter(ans, std::end(ans)),
                [this](int dir) {
                    return env_.dir_to_action(grid::directions[dir]).id();
                }
        );
        return ans;
    }

private:
    CliffWorld env_;
    static const std::unordered_set<int> optimal_actions_[HEIGHT * WIDTH];
};

} // namespace suttonbarto
} // namespace test
} // namespace rl
