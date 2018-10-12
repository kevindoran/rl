#include "gtest/gtest.h"

#include "rl/MappedEnvironment.h"
#include "rl/GridWorld.h"
#include "grid/Grid.h"
#include "rl/IterativePolicyEvaluation.h"
#include "rl/Policy.h"
#include "rl/DeterministicPolicy.h"
#include "common/SuttonBartoExercises.h"
#include "common/ExamplePolicies.h"
#include "rl/RandomGridPolicy.h"

namespace {
// note: We can make GridWord inherit from an abstract class allowing methods to use the interface
// instead of having to become templated to deal with the W & H params.
template<int W, int H>
rl::DeterministicLambdaPolicy create_down_up_policy(const rl::GridWorld<W,H>& grid_world) {
    // note: if the return type is not specified, the action gets returned by value, which
    // leads to an error later on when the reference is used.
    auto fctn = [&grid_world](const rl::Environment& e, const rl::State& s) -> const rl::Action& {
        grid::Position pos = grid_world.state_to_pos(s);
        // We can't just go down, as we will get an exception trying to go outside the grid.
        bool can_go_down = grid_world.grid().is_valid(pos.adj(grid::Direction::DOWN));
        if (can_go_down) {
            return grid_world.dir_to_action(grid::Direction::DOWN);
        } else {
            return grid_world.dir_to_action(grid::Direction::UP);
        }
    };
    return rl::DeterministicLambdaPolicy(fctn);
}

/**
 * Leaving the below here as an example of what _not_ to do.
 * The following Policy is ill-formed as the policy's possible_actions() method will return random
 * results. This method should instead always return the same values for a given state.
 */

template<int W, int H>
rl::DeterministicLambdaPolicy create_random_policy_broken(const rl::GridWorld<W, H>& grid_world) {
    auto fctn = [&grid_world](const rl::Environment& e, const rl::State& s) -> const rl::Action& {
        grid::Position from = grid_world.state_to_pos(s);
        grid::Direction d = grid::random_direction();
        grid::Position to = from.adj(d);
        while(!grid_world.grid().is_valid(to)) {
            d = grid::random_direction();
            to = from.adj(d);
        }
        Ensures(grid_world.grid().is_valid(to));
        return grid_world.dir_to_action(d);
    };
    return rl::DeterministicLambdaPolicy(fctn);
}

rl::MappedEnvironment single_state_action_env(std::string state_name="State 1",
        std::string action_name="Action 1", double reward_value=1.0) {
    rl::MappedEnvironment env;
    const rl::State& state = env.add_state(state_name);
    const rl::Action& action = env.add_action(action_name);
    const rl::Reward& reward = env.add_reward(reward_value, "Reward 1");
    env.add_transition(rl::Transition(state, state, action, reward));
    env.build_distribution_tree();
    return env;
}


} // namespace

TEST(IterativePolicyEvaluationTest, basic_example) {
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
    rl::IterativePolicyEvaluation e;
    rl::DeterministicLambdaPolicy down_up_policy = create_down_up_policy(grid_world);

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

/**
 * This test recreates the square grid and random policy described in exercise 4.1 of
 * (Sutton & Barto, 2018).
 *
 *  E  1  2  3
 *  4  5  6  7
 *  8  9  10 11
 *  12 13 14 E
 *
 *  R = -1, for all transitoins.
 *
 *  The test checks that our policy evaluation routine correctly calculates the value function for
 *  the random policy as being:
 *
 *  0.0  -14  -20  -22
 *  -14  -18  -20  -20
 *  -20  -20  -18  -14
 *  -22  -20  -14  0.0
 */
TEST(IterativePolicyEvaluationTest, sutton_barto_exercise_4_1) {

    rl::IterativePolicyEvaluation evalator;
    const int HEIGHT = 4;
    rl::GridWorld<HEIGHT, HEIGHT> grid_world = rl::test::Exercise4_1::create_grid_world();
    rl::RandomGridPolicy random_policy(grid_world);
    const double allowed_error_factor = 0.02;

    // Test.
    rl::ValueFunction v_fctn = evalator.evaluate(grid_world.environment(), random_policy);
    double expected_values[] =
            {0.0, -14, -20, -22,
             -14, -18, -20, -20,
             -20, -20, -18, -14,
             -22, -20, -14, 0.0};
    for(rl::ID state_id = 0; state_id < grid_world.environment().state_count(); state_id++) {
        ASSERT_NEAR(expected_values[state_id],
                    v_fctn.value(grid_world.environment().state(state_id)),
                    allowed_error_factor * std::abs(expected_values[state_id]));
    }
}

/**
 * This test implements the policy evaluation directly for the example from exercise 4.1 in
 * (Sutton & Barto, 2018).
 *
 * This test was implemented to investigate why the above sutton_barto_exercise_4_1 test was
 * failing (in the end, my GridWorld implementation had different boundary behaviour). I'm leaving
 * the test here, as it is nice to compare it to the above implementation that uses the Policy and
 * Environment abstractions.
 *
 * If the sutton_barto_exercise_4_1 test fails and this one doesn't, the error is probably caused
 * by a faulty policy evaluation routine being used. If both fail, there is likely an issue with
 * Grid. If just this test fails, then this test is probably implemented incorrectly.
 */
TEST(IterativePolicyEvaluationTest, sutton_barto_exercise_4_1_manual) {
    const int HEIGHT = 4;
    const int WIDTH = HEIGHT;
    const grid::Position top_left{0, 0};
    const grid::Position bottom_right{HEIGHT-1, WIDTH-1};
    const int tile_count = HEIGHT*WIDTH;
    const double error_threshold = 0.001;
    // The stopping threshold doesn't correspond to the real error. 0.02 works well for a stopping
    // threshold of 0.001 (a threshold of 0.001 will produces some errors greater than 0.01).
    const double allowed_error_factor = 0.02;
    const double transition_reward = -1.0;
    const double expected_values[] =
            {0.0, -14, -20, -22,
             -14, -18, -20, -20,
             -20, -20, -18, -14,
             -22, -20, -14, 0.0};
    grid::Grid<HEIGHT, WIDTH> grid;
    double ans[tile_count] = {0};

    // Iterative policy evaluation implementation (in-place).
    double error = std::numeric_limits<double>::max();
    while(error > error_threshold) {
        error = 0;
        for(int t = 0; t < tile_count; t++) {
            grid::Position from = grid.to_position(t);
            // Skip the terminal states.
            if(from == top_left || from == bottom_right) {
                continue;
            }
            double reward_sum = 0;
            int action_count = 0;
            for(grid::Direction dir : grid::directions) {
                grid::Position to = from.adj(dir);
                // Transition to same state if going out of bounds.
                if(!grid.is_valid(to)) {
                    to = from;
                }
                action_count++;
                reward_sum += ans[grid.to_id(to)] + transition_reward;
            }
            double value = reward_sum / action_count;
            double prev_value = ans[t];
            ans[t] = value;
            error = std::max(error, std::abs(value - prev_value));
        }
    }

    // Test.
    ASSERT_EQ(sizeof(expected_values), sizeof(ans)) << "The test is faulty if this fails.";
    for(int t = 0; t < tile_count; t++) {
        ASSERT_NEAR(expected_values[t], ans[t],
                    allowed_error_factor * std::abs(expected_values[t]));
    }
}

/**
 * Tests that the policy evaluation correctly converges for a simple continuous task.
 *
 * Tests that the single state, single action environment is evaluated correctly with different
 * discount rates.
 *
 *       state1__
 *        ^      | (action1) reward = 5.
 *        |______|
 *
 *  Tests discount rates from 0.1 to 0.9 (inclusive). Asserts that the value assigned to the single
 *  state is given by:
 *
 *      state_value = reward_value / (1 - discount_rate)
 */
TEST(IterativePolicyEvaluationTest, continuous_task) {
    // Setup
    double reward_value = 5;
    rl::MappedEnvironment env = single_state_action_env("State 1", "Action 1", reward_value);
    const rl::State& state = *env.states_begin();
    rl::IterativePolicyEvaluation evaluation;
    rl::test::FirstActionPolicy policy;

    // Test
    for(int discount_rate_tenth = 1; discount_rate_tenth <= 9; discount_rate_tenth++) {
        double discount_rate = discount_rate_tenth / 10.0;
        double denom = 1 - discount_rate;
        ASSERT_FALSE(denom == 0) << "The test implementation is broken if this fails.";
        double correct_value = reward_value / denom;
        evaluation.set_discount_rate(discount_rate);
        rl::ValueFunction value_fctn = evaluation.evaluate(env, policy);
        // We could be more exact here with out bounds.
        double bounds = 0.01 * correct_value;
        ASSERT_NEAR(correct_value, value_fctn.value(state), bounds);
    }
}

/**
 * An exception should be thrown during policy evaluation if:
 *    1. the policy does not have an action for a state.
 *    2. the policy has an action that has zero weight.
 */
TEST(IterativePolicyEvaluationTest, broken_policy) {
    // Setup
    rl::MappedEnvironment env = single_state_action_env();
    rl::IterativePolicyEvaluation evaluation;
    // Set a discount rate so that the test doesn't go on forever in the case where the behaviour is
    // broken.
    evaluation.set_discount_rate(0.9);
    rl::test::NoActionPolicy no_action_policy;
    rl::test::ZeroWeightActionPolicy zero_weight_policy;

    // Test
    // 1. Exception expected for a policy with no action for a state.
    EXPECT_ANY_THROW(evaluation.evaluate(env, no_action_policy));

    // 2. Exception expected for a policy with an action of zero weight.
    EXPECT_ANY_THROW(evaluation.evaluate(env, zero_weight_policy));
}
