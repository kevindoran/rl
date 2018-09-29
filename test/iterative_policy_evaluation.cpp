#include "gtest/gtest.h"
#include "core/Environment.h"
#include "core/GridWorld.h"
#include "core/Grid.h"
#include "core/IterativePolicyEvaluation.h"
#include "core/Policy.h"


TEST(IterativePolicyEvaluationTest, basic_example) {
    // Setup
    /*
     * Grid world layout:
     *
     *  B  X
     *  X  X
     *  X  X
     *  X  X
     *  E  X
     *
     *  All actions produce a reward of -1.
     *
     */
    const int HEIGHT = 5;
    const int WIDTH = 2;
    rl::GridWorld<HEIGHT, WIDTH> grid_world;
    // Make top-left and bottom-right tiles the end states.
    grid::Position top_left{0, 0};
    grid::Position bottom_left{HEIGHT-1, 0};
    grid_world.environment().mark_as_end_state(grid_world.pos_to_state(bottom_left));
    grid_world.environment().set_start_state(grid_world.pos_to_state(top_left));
    rl::IterativePolicyEvaluation e;
    rl::LambdaPolicy down_up_policy(
        [&grid_world](const rl::Environment& e) {
            grid::Position current = grid_world.state_to_pos(e.current_state().id());
            // We can't just go down, as we will get an exception trying to go outside the grid.
            bool can_go_down = grid_world.grid().is_valid(current.adj(grid::Direction::DOWN));
            if(can_go_down) {
                return grid_world.dir_to_action(grid::Direction::DOWN);
            } else {
               return grid_world.dir_to_action(grid::Direction::UP);
            }
        });

    // Test
    rl::ValueFunction v_fctn = e.evaluate(grid_world.environment(), down_up_policy);
    // With the down-up policy, the state values should be:
    /**
     * -4  inf
     * -3  inf
     * -2  inf
     * -1  inf
     *  0  inf
     *
     *  The right side will oscillate forever. The value will be some high number that depends
     *  on how many iterations the policy evaluation routine carries out. We could have introduced
     *  a move limit, however, we want our grid world to have the property that the visual grids
     *  represent ALL states, and if we have a turn limit, then there are many more states per
     *  position (e.g. (4,3) with 3 turns left, (4,3) with 2 turns left, etc).
     */
    ASSERT_EQ(-4, v_fctn.value(grid_world.pos_to_state(grid::Position{0, 0})));
    ASSERT_EQ(-3, v_fctn.value(grid_world.pos_to_state(grid::Position{0, 1})));
    ASSERT_EQ(-2, v_fctn.value(grid_world.pos_to_state(grid::Position{0, 2})));
    ASSERT_EQ(-1, v_fctn.value(grid_world.pos_to_state(grid::Position{0, 3})));
    ASSERT_EQ(0, v_fctn.value(grid_world.pos_to_state(grid::Position{0, 4})));
}
