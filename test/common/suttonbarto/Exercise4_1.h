#pragma once

#include "TestEnvironment.h"
#include "rl/GridWorld.h"
#include "grid/Grid.h"

namespace rl {
namespace test {
namespace suttonbarto {

/**
 * This test case recreates the square grid and random policy described in exercise 4.1 of
 * (Sutton & Barto, 2018).
 *
 *  E  1  2  3
 *  4  5  6  7
 *  8  9  10 11
 *  12 13 14 E
 *
 *  R = -1, for all transitions.
 *
 *  The test checks that our policy evaluation routine correctly calculates the value function for
 *  the random policy as being:
 *
 *  0.0  -14  -20  -22
 *  -14  -18  -20  -20
 *  -20  -20  -18  -14
 *  -22  -20  -14  0.0
 */
class Exercise4_1 : public TestEnvironment {
public:
    static const int GRID_WIDTH = 4;
    static const int GRID_HEIGHT = 4;
    using GridType = GridWorld<GRID_HEIGHT, GRID_WIDTH>;

    static constexpr double expected_values[] =
            {0.0, -14, -20, -22,
             -14, -18, -20, -20,
             -20, -20, -18, -14,
             -22, -20, -14, 0.0};

    std::string name() const override {
        return "Sutton & Barto exercise 4.1";
    }

    double required_discount_rate() const override {
        return 1.0;
    }

    double required_delta_threshold() const override {
        return 1e-2;
    }

    const Environment& env() const override {
        return grid_world_;
    }

    OptimalActions optimal_actions(const State& from_state) const override {
        OptimalActions ans;
        std::transform(
                std::begin(optimal_actions_[from_state.id()]),
                std::end(optimal_actions_[from_state.id()]),
                std::inserter(ans, std::end(ans)),
                [this](int dir) {
                    return grid_world_.dir_to_action(grid::directions[dir]).id();
                }
        );
        return ans;
    }

private:
    static GridType create_grid_world() {
        // Setup.
        rl::GridWorld<GRID_HEIGHT, GRID_WIDTH> grid_world;
        const grid::Position top_left{0, 0};
        const grid::Position bottom_right{GRID_HEIGHT - 1, GRID_WIDTH - 1};
        grid_world.mark_as_end_state(grid_world.pos_to_state(top_left));
        grid_world.mark_as_end_state(grid_world.pos_to_state(bottom_right));
        grid_world.set_all_rewards_to(-1.0);
        return grid_world;
    }

private:
    GridType grid_world_ = create_grid_world();

    static const std::unordered_set<int> optimal_actions_[GRID_WIDTH * GRID_HEIGHT];
};

} // namespace suttonbarto
} // namespace test
} // namespace rl

