#include <rl/FirstVisitMCValuePredictor.h>
#include "gtest/gtest.h"

#include "rl/Environment.h"
#include "rl/Policy.h"
#include "common/PolicyEvaluationTests.h"
#include "rl/IterativePolicyEvaluator.h"

class IterativePolicyEvaluator : public ::testing::Test {
protected:
    rl::IterativePolicyEvaluator evaluator;
};

TEST_F(IterativePolicyEvaluator, grid_world1) {
    // Setup
    rl::test::GridWorldTest1 test_case;
    // Test
    test_evaluator(evaluator, test_case);
}

TEST_F(IterativePolicyEvaluator, sutton_barto_exercise_4_1) {
    // Setup
    rl::test::SuttonBartoExercise4_1 test_case;
    // Test
    test_evaluator(evaluator, test_case);
}

TEST_F(IterativePolicyEvaluator, continuous_task) {
    // Avoid using double in a loop condition.
    for(int discount_rate_tenth = 1; discount_rate_tenth <= 9; discount_rate_tenth++) {
        // Setup
        double discount_rate = discount_rate_tenth / 10.0;
        rl::test::ContinuousTaskTest test_case(discount_rate);
        evaluator.set_discount_rate(discount_rate);
        // Test
        test_evaluator(evaluator, test_case);
    }
}

TEST(FirstVisitMCValueBasedEvaluator, grid_world1) {
    // Setup
    rl::FirstVisitMCValuePredictor evaluator;
    rl::test::GridWorldTest1 test_case;

    // Test
    test_evaluator(evaluator, test_case);
}

TEST(FirstVisitMCValueBasedEvaluator, sutton_barto_exercise_4_1) {
    // Setup
    rl::FirstVisitMCValuePredictor evaluator;
    // The default (currently 0.00001) leads to long execution times. Making it less strict.
    evaluator.set_delta_threshold(0.0001);
    rl::test::SuttonBartoExercise4_1 test_case;

    // Test
    test_evaluator(evaluator, test_case);
}