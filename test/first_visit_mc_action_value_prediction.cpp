#include "gtest/gtest.h"

#include "rl/GridWorld.h"
#include "rl/Policy.h"
#include "rl/FirstVisitMCActionValuePredictor.h"
#include "common/ExamplePolicies.h"
#include "rl/RandomGridPolicy.h"
#include "common/SuttonBartoExercises.h"


TEST(FirstVisitMCActionValuePrediction, basic_example) {
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
    rl::FirstVisitMCActionValuePredictor evaluator;
    rl::DeterministicLambdaPolicy down_up_policy = rl::test::create_down_up_policy(grid_world);

    // Test
    const rl::ActionValueFunction& action_v_fctn =
            evaluate(evaluator, grid_world.environment(), down_up_policy);
    // With the down-up policy, the action values should be:
    /**
     * Down  Up  Left  Right
     * -4    -5   -5    -5
     * -3    -5   -4    -4
     * -2    -4   -3    -3
     * -1    -3   -2    -2
     * NA    NA   NA    NA
     *
     *  Rules behind the values:
     *  1. The state-Down values are the same as the state values for the state value function.
     *  2. The state-Up values are the same as the sate values for the _above_ state value function.
     *     The exception being 0-Up, as there is no state north of the 0 state.
     *  3. The grid boundary behaviour is TRANSITION_TO_CURRENT, so movements directed outside the
     *     grid cause no change in state. Thus, state-no_movement_action will have a value equal
     *     to the state value -1 (-1 reward, but no state change).
     *  4. For state-action pairs that are not valid, NA is listed. For this test, we will assert
     *     that the NA values be represented by zero. This aspect of the API still needs some
     *     clarification.
     */
    double expected_action_values[][grid::DIR_COUNT] =
          // In ordinal order.
          //  0     1    2    3
          // Right  Down Left Up
            { {-5,  -4,  -5,  -5},
              {-4,  -3,  -4,  -5},
              {-3,  -2,  -3,  -4},
              {-2,  -1,  -2,  -3},
              { 0,   0,   0,   0} };
    for(int h = 0; h < HEIGHT; h++) {
        for(grid::Direction d : grid::directions) {
            int direction_ordinal = static_cast<int>(d);
            double correct_value = expected_action_values[h][direction_ordinal];
            grid::Position pos{h, 0};
            const rl::Action& action = grid_world.dir_to_action(d);
            double value_to_test = action_v_fctn.value(grid_world.pos_to_state(pos), action);
            ASSERT_EQ(correct_value, value_to_test);
        }
    }
}

TEST(FirstVisitMCActionValuePrediction, exercise5_1_specific_case) {
    // Setup
    using BlackjackEnv = rl::test::Exercise5_1::BlackjackEnvironment;
    BlackjackEnv env;
    const BlackjackEnv::BlackjackState start_state{15, false, 2};
    rl::DeterministicLambdaPolicy hit_then_stick(
            [&env, start_state](const rl::Environment&, const rl::State& state) -> const rl::Action& {
                if(start_state == env.blackjack_state(state)) {
                    return env.action(env.action_id(BlackjackEnv::BlackjackAction::HIT));
                } else {
                    return env.action(env.action_id(BlackjackEnv::BlackjackAction::STICK));
                }
            }
    );
    rl::FirstVisitMCActionValuePredictor predictor;
    // FIXME: what should be done to get this threshold down?
    double allowed_error = 0.03;

    // Test
    const rl::ActionValueFunction& value_fctn = rl::evaluate(predictor, env, hit_then_stick);
    // From BlackjackEnvironmentF.test_specific_case_3, we know that the expected return from
    // (15, false, 2) with the hit-stick policy is:
    double expected_return = 0.267040 - 0.683266; // wins - losses
    ASSERT_NEAR( expected_return,
        value_fctn.value(env.state(start_state), env.action(BlackjackEnv::BlackjackAction::HIT)),
        allowed_error);
}
