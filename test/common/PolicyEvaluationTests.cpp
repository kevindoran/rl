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

void rl::test::test_evaluator(rl::StateBasedEvaluator& evaluator,
                              const rl::test::StateBasedEvaluatorTestCase& test_case) {
    test_case.check(evaluate(evaluator, test_case.env(), test_case.policy()));
}

void rl::test::test_evaluator(rl::ActionBasedEvaluator& evaluator,
                              const rl::test::ActionBasedEvaluatorTestCase& test_case) {
    test_case.check(evaluate(evaluator, test_case.env(), test_case.policy()));
}

rl::test::GridWorldTest1::GridWorldTest1() {
    grid::Position top_left{0, 0};
    grid::Position bottom_left{HEIGHT-1, 0};
    grid_world.environment().set_start_state(grid_world.pos_to_state(top_left));
    grid_world.environment().mark_as_end_state(grid_world.pos_to_state(bottom_left));
    grid_world.environment().set_all_rewards_to(-1.0);
    grid_world.environment().build_distribution_tree();
    // Assumption here is that DeterministicLambdaPolicy is copy constructable.
    p_down_up_policy = std::make_unique<rl::DeterministicLambdaPolicy>(
            rl::test::create_down_up_policy(grid_world));
}

void rl::test::GridWorldTest1::check(const rl::ValueFunction& value_function) const {
    ASSERT_EQ(-4, value_function.value(grid_world.pos_to_state(grid::Position{0, 0})));
    ASSERT_EQ(-3, value_function.value(grid_world.pos_to_state(grid::Position{0, 1})));
    ASSERT_EQ(-2, value_function.value(grid_world.pos_to_state(grid::Position{0, 2})));
    ASSERT_EQ(-1, value_function.value(grid_world.pos_to_state(grid::Position{0, 3})));
    ASSERT_EQ(0,  value_function.value(grid_world.pos_to_state(grid::Position{0, 4})));
}

void rl::test::SuttonBartoExercise4_1::check(const rl::ValueFunction& value_function) const {
    for(rl::ID state_id = 0; state_id < test_case.env().state_count(); state_id++) {
        ASSERT_NEAR(Ex4_1::expected_values[state_id],
                    value_function.value(test_case.env().state(state_id)),
                    ALLOWED_ERROR_FACTOR * std::abs(Ex4_1::expected_values[state_id]));
    }
}

rl::test::ContinuousTaskTest::ContinuousTaskTest(double discount_rate) :
discount_rate(discount_rate),
env_(single_state_action_env("State 1", "Action 1", REWARD_VALUE)) {
}

void rl::test::ContinuousTaskTest::check(const rl::ValueFunction& value_function) const {
    double denom = 1.0 - discount_rate;
    ASSERT_FALSE(denom == 0) << "The test implementation is broken if this fails.";
    double correct_value = REWARD_VALUE / denom;
    // We could be more exact here with out bounds.
    double bounds = ALLOWED_ERROR_FACTOR * correct_value;
    const State& the_only_state = *env_.states().begin();
    ASSERT_NEAR(correct_value, value_function.value(the_only_state), bounds);
}
