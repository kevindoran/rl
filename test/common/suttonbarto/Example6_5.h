#pragma once

#include "rl/GridWorld.h"
#include "rl/Trial.h"

namespace rl {
namespace test {
namespace suttonbarto {

// Example6_5 doesn't extend TestEnvironment as the example doesn't provide the full optimal
// policy (only the path taken by the optimal policy).
/**
 * Refer to p130 of Sutton & Barto 2018.
 */
class Example6_5 {
public:
    static constexpr int HEIGHT = 7;
    static constexpr int WIDTH = 10;
    using GridType = GridWorld<HEIGHT, WIDTH>::GridType;
    static constexpr double TRANSITION_REWARD = -1;
    static constexpr grid::Position GOAL_POS{3,7};
    static constexpr grid::Position START_POS{3,0};
    static constexpr int WIND_STRENGTH[WIDTH] =
    // column: 0  1  2  3  4  5  6  7  8  9
              {0, 0, 0, 1, 1, 1, 2, 2, 1, 0};
    class WindyGridWorld : public GridWorld<HEIGHT, WIDTH> {
    public:
        WindyGridWorld() {
            set_all_rewards_to(TRANSITION_REWARD);
            mark_as_end_state(pos_to_state(GOAL_POS));
            set_start_state(pos_to_state(START_POS));
        }

        Response next_state(const State& from_state, const Action& action) const override {
            const State& after_wind = apply_wind(from_state);
            return GridWorld::next_state(after_wind, action);
        }

    private:
        const State& apply_wind(const State& state) const {
            grid::Position pos = state_to_pos(state);
            int wind_strength = WIND_STRENGTH[pos.x];
            grid::Position after_wind = pos.translate(grid::Direction::UP, wind_strength);
            // Make sure the position isn't outside the grid.
            after_wind = GridType::round(after_wind);
            return pos_to_state(after_wind);
        }
    };
    static const std::vector<grid::Position> OPTIMAL_ROUTE;
};

} // namespace suttonbarto
} // namespace test
} // namespace rl
