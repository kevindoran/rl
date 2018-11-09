#include "gtest/gtest.h"

#include "rl/Environment.h"
#include "rl/Policy.h"
#include "common/PolicyEvaluationTests.h"
#include "rl/IterativePolicyEvaluator.h"
#include "rl/FirstVisitMCValuePredictor.h"
#include "rl/FirstVisitMCActionValuePredictor.h"

//----------------------------------------------------------------------------------------------
// IterativePolicyEvaluator
//----------------------------------------------------------------------------------------------
class IterativePolicyEvaluator : public ::testing::Test {
protected:
    rl::IterativePolicyEvaluator evaluator;
};

TEST_F(IterativePolicyEvaluator, grid_world1) {
    // Setup
    rl::test::GridWorldTest1 test_case;
    // Test
    test_case.check(evaluator);
}

TEST_F(IterativePolicyEvaluator, sutton_barto_exercise_4_1) {
    // Setup
    rl::test::SuttonBartoExercise4_1Test test_case;
    // Test
    test_case.check(evaluator);
}

TEST_F(IterativePolicyEvaluator, continuous_task) {
    rl::test::ContinuousTaskTest test_case;
    test_case.check(evaluator);
}

TEST_F(IterativePolicyEvaluator, broken_policy) {
    rl::test::BrokenPolicyTest test_case;
    test_case.check(evaluator);
}

//----------------------------------------------------------------------------------------------
// First-visit Monte Carlo state value function evaluator.
//----------------------------------------------------------------------------------------------
class FirstVisitMCValuePredictor : public ::testing::Test {
protected:
    rl::FirstVisitMCValuePredictor evaluator;
};

TEST_F(FirstVisitMCValuePredictor, grid_world1) {
    // Setup
    rl::test::GridWorldTest1 test_case;
    // Test
    test_case.check(evaluator);
}

TEST_F(FirstVisitMCValuePredictor, sutton_barto_exercise_4_1) {
    // Setup
    // The default (currently 0.00001) leads to long execution times. Making it less strict.
    evaluator.set_delta_threshold(0.0001);
    rl::test::SuttonBartoExercise4_1Test test_case;
    // Test
    test_case.check(evaluator);
}

//----------------------------------------------------------------------------------------------
// First-visit Monte Carlo state value function evaluator.
//----------------------------------------------------------------------------------------------

class FirstVisitMCActionValuePredictor : public ::testing::Test {
protected:
    rl::FirstVisitMCActionValuePredictor evaluator;
};

TEST_F(FirstVisitMCActionValuePredictor, grid_world1) {
    // Setup
    rl::test::GridWorldTest1 test_case;
    test_case.check(evaluator);
}

TEST_F(FirstVisitMCActionValuePredictor,
        blackjack_speciic_case1_LONG_RUNNING) {
    // Setup
    rl::test::BlackjackSpecificCase test_case;
    test_case.check(evaluator);
}

