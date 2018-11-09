#include "PolicyEvaluationTests.h"

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
