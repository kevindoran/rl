#include "gtest/gtest.h"
#include "rl/GridWorld.h"
#include "rl/Policy.h"
#include "rl/FirstVisitMCValuePrediction.h"
#include "common/ExamplePolicies.h"
#include "rl/RandomGridPolicy.h"
#include "common/SuttonBartoExercises.h"

// TODO: the next two tests were copied directly from iterative_policy_evaluation.cpp. It would be
// nice to create a more general testing setup such that multiple evaluation methods can be tested
// easily with the same data.
TEST(FirstVisitMCValuePredictionTest, basic_example) {
    // Setup
    /*
     * Grid world layout:
     *
     *  X
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
    rl::GridWorld<HEIGHT, WIDTH> grid_world(rl::GridWorldBoundsBehaviour::TRANSITION_TO_CURRENT);
    // Make bottom-right tile the end state.
    grid::Position top_left{0, 0};
    grid::Position bottom_left{HEIGHT-1, 0};
    grid_world.environment().mark_as_end_state(grid_world.pos_to_state(bottom_left));
    grid_world.environment().set_all_rewards_to(-1.0);
    grid_world.environment().build_distribution_tree();
    rl::FirstVisitMCValuePrediction evaluator;
    rl::DeterministicLambdaPolicy down_up_policy = rl::test::create_down_up_policy(grid_world);

    // Test
    const rl::ValueFunction& v_fctn = evaluate(evaluator, grid_world.environment(), down_up_policy);
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

/**
 * This test recreates the square grid and random policy described in exercise 4.1 of
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

TEST(FirstVisitMCValuePredictionTest, sutton_barto_exercise_4_1) {
    rl::test::Exercise4_1 test_case;
    rl::FirstVisitMCValuePrediction evaluator;
    // The default (currently 0.00001) leads to long execution times. Making it less strict.
    evaluator.set_delta_threshold(0.0001);
    auto& grid_world = test_case.grid_world();
    rl::RandomGridPolicy random_policy(grid_world);
    const double allowed_error_factor = 0.1;

    // Test.
    const rl::ValueFunction& v_fctn = evaluate(evaluator, grid_world.environment(), random_policy);
    for(rl::ID state_id = 0; state_id < grid_world.environment().state_count(); state_id++) {
        ASSERT_NEAR(test_case.expected_values[state_id],
                    v_fctn.value(grid_world.environment().state(state_id)),
                    allowed_error_factor * std::abs(test_case.expected_values[state_id]));
    }
}