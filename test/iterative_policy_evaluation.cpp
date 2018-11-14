#include "gtest/gtest.h"

#include "rl/MappedEnvironment.h"
#include "rl/GridWorld.h"
#include "grid/Grid.h"
#include "rl/IterativePolicyEvaluator.h"
#include "rl/Policy.h"
#include "rl/DeterministicPolicy.h"
#include "common/suttonbarto/Exercise4_1.h"
#include "common/ExamplePolicies.h"
#include "rl/RandomPolicy.h"

namespace {

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

} // namespace

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
    using TestInfo = rl::test::suttonbarto::Exercise4_1;
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
    ASSERT_EQ(sizeof(TestInfo::expected_values), sizeof(ans))
        << "The test is faulty if this fails.";
    for(int t = 0; t < tile_count; t++) {
        ASSERT_NEAR(TestInfo::expected_values[t], ans[t],
                    allowed_error_factor * std::abs(TestInfo::expected_values[t]));
    }
}
