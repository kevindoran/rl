#pragma once

#include "rl/GridWorld.h"
#include "rl/Trial.h"

namespace rl {
namespace test {
namespace suttonbarto {

static constexpr int WINDY_GRID_HEIGHT = 7;
static constexpr int WINDY_GRID_WIDTH = 10;
class WindyGridWorld : public GridWorld<WINDY_GRID_HEIGHT, WINDY_GRID_WIDTH> {
public:
    using GridType = GridWorld<WINDY_GRID_HEIGHT, WINDY_GRID_WIDTH>::GridType;
    static constexpr double TRANSITION_REWARD = -1;
    static constexpr grid::Position GOAL_POS{3,7};
    static constexpr grid::Position START_POS{3,0};
    static constexpr int WIND_STRENGTH[WINDY_GRID_WIDTH] =
    // column: 0  1  2  3  4  5  6  7  8  9
              {0, 0, 0, 1, 1, 1, 2, 2, 1, 0};
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

} // namespace suttonbarto
} // namespace test
} // namespace rl


