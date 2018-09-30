#include "gtest/gtest.h"
#include "core/Environment.h"
#include "core/GridWorld.h"

using namespace rl;

TEST(GridWorldTest, basic_example) {
    // Setup
    /*
     *
     *  E  X  X  X
     *  X  X  X  X
     *  X  X  X  X
     *  X  X  X  E
     *
     */
    const int HEIGHT = 4;
    const int WIDTH = 4;
    GridWorld<HEIGHT, WIDTH> grid_world;
    Environment& environment = grid_world.environment();
    auto& grid = grid_world.grid();
    // Make top-left and bottom-right tiles the end states.
    grid::Position top_left{0, 0};
    grid::Position bottom_right{HEIGHT-1, WIDTH-1};
    environment.mark_as_end_state(grid_world.pos_to_state(top_left));
    environment.mark_as_end_state(grid_world.pos_to_state(bottom_right));

    // Test
    // 1. Start at (0, 1) and move to the right edge.
    grid::Position pos{0, 1};
    environment.set_start_state(grid_world.pos_to_state(pos));
    const Action& move_right_action = grid_world.dir_to_action(grid::Direction::RIGHT);
    while(pos.x < WIDTH-1) {
        environment.execute_action(move_right_action);
        pos = pos.adj(grid::Direction::RIGHT);
        ASSERT_EQ(pos, grid_world.current_pos());
        ASSERT_EQ(0, environment.accumulated_reward())
            << "The rewards should all be zero by default.";
    }

    // 2. An exception should be thrown trying to move off the grid.
    ASSERT_ANY_THROW(environment.execute_action(move_right_action));

    // 3. Set all rewards to 1.0. Then move down.
    for(int t = 0; t < grid.TILE_COUNT; t++) {
        ID reward_at_t = grid_world.reward_id(grid.to_position(t));
        environment.reward(reward_at_t).set_value(1.0);
    }
    const Action& move_down_action = grid_world.dir_to_action(grid::Direction::DOWN);
    environment.execute_action(move_down_action);
    pos = pos.adj(grid::Direction::DOWN);
    ASSERT_EQ(pos, grid_world.current_pos()) << "We should have moved down by 1.";
    ASSERT_EQ(1.0, environment.accumulated_reward()) << "We should now have 1.0 rewarded.";
}