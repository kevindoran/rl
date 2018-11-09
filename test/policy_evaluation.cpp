#include "gtest/gtest.h"

#include "rl/Environment.h"
#include "rl/Policy.h"
#include "common/PolicyEvaluationTests.h"
#include "rl/IterativePolicyEvaluator.h"


TEST(IterativePolicyEvaluation, grid_world1) {
    // Setup
    rl::IterativePolicyEvaluator evaluator;
    rl::test::GridWorldTest1 test_case;

    // Test
    test_evaluator(evaluator, test_case);
}