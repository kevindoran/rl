#include "gtest/gtest.h"
#include "rl/GridWorld.h"
#include "rl/Trial.h"

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
    GridWorld<HEIGHT, WIDTH> grid_world(GridWorldBoundsBehaviour::NO_OUT_OF_BOUNDS);
    grid::Position pos{0, 1};
    grid_world.set_start_state(grid_world.pos_to_state(pos));
    rl::Trial grid_trial(grid_world);
    // Make top-left and bottom-right tiles the end states.
    grid::Position top_left{0, 0};
    grid::Position bottom_right{HEIGHT-1, WIDTH-1};
    grid_world.mark_as_end_state(grid_world.pos_to_state(top_left));
    grid_world.mark_as_end_state(grid_world.pos_to_state(bottom_right));

    // Test
    // 1. Start at (0, 1) and move to the right edge.
    const Action& move_right_action = grid_world.dir_to_action(grid::Direction::RIGHT);
    while(pos.x < WIDTH-1) {
        ASSERT_TRUE(grid_world.is_action_allowed(grid_world.pos_to_state(pos), move_right_action));
        grid_trial.execute_action(move_right_action);
        pos = pos.adj(grid::Direction::RIGHT);
        ASSERT_EQ(pos, grid_world.state_to_pos(grid_trial.current_state()));
        ASSERT_EQ(0, grid_trial.accumulated_reward())
            << "The rewards should all be zero by default.";
    }

    // 2. An exception should be thrown trying to move off the grid.
    //    This occurs due to our NO_OUT_OF_BOUNDS option.
    ASSERT_ANY_THROW(grid_trial.execute_action(move_right_action));

    // 3. Set all rewards to 1.0. Then move down.
    grid_world.set_all_rewards_to(1.0);
    const Action& move_down_action = grid_world.dir_to_action(grid::Direction::DOWN);
    grid_trial.execute_action(move_down_action);
    pos = pos.adj(grid::Direction::DOWN);
    ASSERT_EQ(pos, grid_world.state_to_pos(grid_trial.current_state()))
        << "We should have moved down by 1.";
    ASSERT_EQ(1.0, grid_trial.accumulated_reward()) << "We should now have 1.0 rewarded.";
}