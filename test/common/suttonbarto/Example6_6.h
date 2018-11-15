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
 * The CliffWorld environment is used to highlight a difference in the behaviour of Q-Learning and
 * Sarsa algorithms: Q-Learning will find the optimal route even though the route is alone by the
 * cliff (which has a very high negative reward). Sarsa with e-greedy will take a safer but not
 * the optimal route. e-greedy Sarsa under-rates the optimal route as the action selection is
 * sometimes random and will follow a route off the cliff. Q-Learning, being an off-policy
 * algorithm, doesn't have this problem.
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
        // 3 cases.
        // Start state:    up
        // Last column:    down
        // Otherwise:      right
        grid::Direction optimal_dir = grid::Direction::NONE;
        if(from_state == env_.start_state()) {
            optimal_dir = grid::Direction::UP;
        } else if(env_.state_to_pos(from_state).x == (WIDTH - 1)) {
            optimal_dir = grid::Direction::DOWN;
        } else {
            optimal_dir = grid::Direction::RIGHT;
        }
        CHECK(optimal_dir != grid::Direction::NONE);
        const Action& optimal_action = env_.dir_to_action(optimal_dir);
        CHECK(env_.is_action_allowed(from_state, optimal_action));
        return OptimalActions{optimal_action.id()};
    }

    static const std::vector<grid::Position> SAFE_ROUTE;

private:
    CliffWorld env_;
};

} // namespace suttonbarto
} // namespace test
} // namespace rl
