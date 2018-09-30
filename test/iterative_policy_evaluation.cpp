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
     *  B
     *  X
     *  X
     *  X
     *  E
     *
     *  All actions produce a reward of -1.
     *
     */
    const int HEIGHT = 5;
    const int WIDTH = 1;
    rl::GridWorld<HEIGHT, WIDTH> grid_world;
    // Make bottom-right tile the end state.
    grid::Position top_left{0, 0};
    grid::Position bottom_left{HEIGHT-1, 0};
    grid_world.environment().mark_as_end_state(grid_world.pos_to_state(bottom_left));
    grid_world.environment().set_start_state(grid_world.pos_to_state(top_left));
    grid_world.environment().set_all_rewards_to(-1.0);
    grid_world.environment().build_distribution_tree();
    rl::IterativePolicyEvaluation e;
    rl::DeterministicLambdaPolicy down_up_policy(
        // note: if the return type is not specified, the action gets returned by value, which
        // leads to an error later on when the reference is used.
        [&grid_world](const rl::Environment& e, const rl::State& s) -> const rl::Action& {
            grid::Position pos = grid_world.state_to_pos(s);
            // We can't just go down, as we will get an exception trying to go outside the grid.
            bool can_go_down = grid_world.grid().is_valid(pos.adj(grid::Direction::DOWN));
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
     * -4
     * -3
     * -2
     * -1
     *  0
     */
    ASSERT_EQ(-4, v_fctn.value(grid_world.pos_to_state(grid::Position{0, 0})));
    ASSERT_EQ(-3, v_fctn.value(grid_world.pos_to_state(grid::Position{0, 1})));
    ASSERT_EQ(-2, v_fctn.value(grid_world.pos_to_state(grid::Position{0, 2})));
    ASSERT_EQ(-1, v_fctn.value(grid_world.pos_to_state(grid::Position{0, 3})));
    ASSERT_EQ(0, v_fctn.value(grid_world.pos_to_state(grid::Position{0, 4})));
}
