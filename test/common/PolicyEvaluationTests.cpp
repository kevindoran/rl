#include "PolicyEvaluationTests.h"

namespace {

rl::MappedEnvironment single_state_action_env(
        const std::string& state_name="State 1", const std::string& action_name="Action 1",
        double reward_value=1.0) {
    rl::MappedEnvironment env;
    const rl::State& state = env.add_state(state_name);
    const rl::Action& action = env.add_action(action_name);
    const rl::Reward& reward = env.add_reward(reward_value, "Reward 1");
    env.add_transition(rl::Transition(state, state, action, reward));
    env.build_distribution_tree();
    return env;
}

} // namespace

namespace rl {
namespace test {

void rl::test::GridWorldTest1::check(StateBasedEvaluator& evaluator) const {
    // Setup
    static const int HEIGHT = 5;
    static const int WIDTH = 1;
    rl::GridWorld<HEIGHT, WIDTH> grid_world{GridWorldBoundsBehaviour::TRANSITION_TO_CURRENT};
    grid::Position top_left{0, 0};
    grid::Position bottom_left{HEIGHT - 1, 0};
    grid_world.environment().set_start_state(grid_world.pos_to_state(top_left));
    grid_world.environment().mark_as_end_state(grid_world.pos_to_state(bottom_left));
    grid_world.environment().set_all_rewards_to(-1.0);
    grid_world.environment().build_distribution_tree();
    DeterministicLambdaPolicy down_up_policy = create_down_up_policy(grid_world);

    // Test
    const ValueFunction& value_function =
            evaluate(evaluator, grid_world.environment(), down_up_policy);
    ASSERT_EQ(-4, value_function.value(grid_world.pos_to_state(grid::Position{0, 0})));
    ASSERT_EQ(-3, value_function.value(grid_world.pos_to_state(grid::Position{0, 1})));
    ASSERT_EQ(-2, value_function.value(grid_world.pos_to_state(grid::Position{0, 2})));
    ASSERT_EQ(-1, value_function.value(grid_world.pos_to_state(grid::Position{0, 3})));
    ASSERT_EQ(0, value_function.value(grid_world.pos_to_state(grid::Position{0, 4})));
}

void SuttonBartoExercise4_1Test::check(StateBasedEvaluator& evaluator) const {
    // Setup
    using Ex4_1 = Exercise4_1;
    Ex4_1 test_case;
    RandomPolicy policy;

    // Test
    const ValueFunction& value_function = evaluate(evaluator, test_case.env(), policy);
    for (rl::ID state_id = 0; state_id < test_case.env().state_count(); state_id++) {
        ASSERT_NEAR(Ex4_1::expected_values[state_id],
                    value_function.value(test_case.env().state(state_id)),
                    ALLOWED_ERROR_FACTOR * std::abs(Ex4_1::expected_values[state_id]));
    }
}

void rl::test::ContinuousTaskTest::check(StateBasedEvaluator& evaluator) const {
    const int REWARD_VALUE = 5;
    MappedEnvironment env(single_state_action_env("State 1", "Action 1", REWARD_VALUE));
    FirstActionPolicy policy;
    for(int discount_rate_tenth = 1; discount_rate_tenth <= 9; discount_rate_tenth++) {
        double discount_rate = discount_rate_tenth / 10.0;
        evaluator.set_discount_rate(discount_rate);
        double denom = 1.0 - discount_rate;
        ASSERT_FALSE(denom == 0) << "The test implementation is broken if this fails.";
        double correct_value = REWARD_VALUE / denom;
        // We could be more exact here with out bounds.
        double bounds = ALLOWED_ERROR_FACTOR * correct_value;
        const State& the_only_state = *env.states().begin();
        const ValueFunction& value_function = evaluate(evaluator, env, policy);
        ASSERT_NEAR(correct_value, value_function.value(the_only_state), bounds);
    }
}

void BrokenPolicyTest::check(StateBasedEvaluator& evaluator) const {
    // Setup
    MappedEnvironment env = single_state_action_env();
    // Set a discount rate so that the test doesn't go on forever in the case where the behaviour is
    // broken.
    evaluator.set_discount_rate(0.9);
    rl::test::NoActionPolicy no_action_policy;
    rl::test::ZeroWeightActionPolicy zero_weight_policy;

    // Test
    // 1. Exception expected for a policy with no action for a state.
    evaluator.initialize(env, no_action_policy);
    EXPECT_ANY_THROW(evaluator.run());

    // 2. Exception expected for a policy with an action of zero weight.
    evaluator.initialize(env, zero_weight_policy);
    EXPECT_ANY_THROW(evaluator.run());
}

} // namespace test
} // namespace rl