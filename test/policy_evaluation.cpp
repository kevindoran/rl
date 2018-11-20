#include <rl/GradientMCLinear.h>
#include <suttonbarto/RandomWalk.h>
#include "gtest/gtest.h"

#include "rl/Environment.h"
#include "rl/Policy.h"
#include "common/PolicyEvaluationTests.h"
#include "rl/IterativePolicyEvaluator.h"
#include "rl/FirstVisitMCValuePredictor.h"
#include "rl/FirstVisitMCActionValuePredictor.h"
#include "rl/MCEvaluator3.h"
#include "rl/TDEvaluator.h"

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

TEST_F(FirstVisitMCValuePredictor, sutton_barto_exercise_4_1_LONG_RUNNING) {
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
    // Test
    test_case.check(evaluator);
}

TEST_F(FirstVisitMCActionValuePredictor,
        blackjack_specific_case1_LONG_RUNNING) {
    // Setup
    rl::test::BlackjackSpecificCase test_case;
    // Another slow test. Reducing the delta threshold to speed it up. (1e-3 leads to a failing test
    // at the current allowed accuracy bounds, so 1e-4 will have to do).
    evaluator.set_delta_threshold(1e-4);
    // Test
    test_case.check(evaluator);
}

//----------------------------------------------------------------------------------------------
// Every-visit Monte Carlo off-policy importance sampling state-action value function evaluator.
//----------------------------------------------------------------------------------------------
class MCEvaluator3 : public ::testing::Test {
protected:
    rl::MCEvaluator3 evaluator;
};

TEST_F(MCEvaluator3, grid_world1) {
    // Setup
    rl::test::GridWorldTest1 test_case;
    test_case.check(evaluator);
}

TEST_F(MCEvaluator3, blackjack_specific_case1_LONG_RUNNING) {
    // Setup
    rl::test::BlackjackSpecificCase test_case;
    // The MCEvaluator3 takes a LONG time to converge for all state-action pairs. Reducing the
    // delta threshold to speed things up. It is lucky that it is still accurate enough to pass
    // with a 1e-3 delta threshold setting.
    evaluator.set_delta_threshold(1e-3);
    // Test
    test_case.check(evaluator);
}

//----------------------------------------------------------------------------------------------
// On-policy temporal difference evaluator.
//----------------------------------------------------------------------------------------------
class TDEvaluator : public ::testing::Test {
protected:
    rl::TDEvaluator evaluator;
};

TEST_F(TDEvaluator, grid_world1) {
    // Setup
    rl::test::GridWorldTest1 test_case;
    // Test
    test_case.check(evaluator);
}

TEST_F(TDEvaluator, blackjack_specific_case1) {
    // Setup
    rl::test::BlackjackSpecificCase test_case;
    evaluator.set_delta_threshold(1e-4);
    // Test
    test_case.check(evaluator);
}

//----------------------------------------------------------------------------------------------
// On-policy Monte-Carlo gradient descent.
//----------------------------------------------------------------------------------------------
TEST(GradientMCLinear, random_walk_1000) {
    // Setup
    rl::GradientMCLinear evaluator;
    rl::test::suttonbarto::RandomWalk1000 env;
    rl::test::FirstValidActionPolicy policy;
    const double max_error = 0.2;
    const int inner_state_count = 1000; // 2 end states.
    const int group_count = 10;
    const int states_per_group = inner_state_count / group_count;
    // +1 as the first state, at 0, is the left terminal state.
    std::vector<int> state_to_group_mapping(inner_state_count + 1);
    // Give all the inner states a group id.
    for(int i = 0; i < inner_state_count; i++) {
        int group = i / states_per_group;
        rl::ID state_id = i + 1;
        state_to_group_mapping[state_id] = group;
    }
    rl::StateAggregateValueFunction value_function(group_count, state_to_group_mapping);
    // Any policy will do, as the RandomWalk environment ignores the policy.
    // Calculate expected result via iterative DP.
    rl::IterativePolicyEvaluator iterative_evaluator;
    iterative_evaluator.set_delta_threshold(1e-3);
    rl::ValueTable expected = rl::evaluate(iterative_evaluator, env, policy);

    // Test
    // Calculate the state-aggregated value function.
    evaluator.evaluate(env, policy, value_function);
    for(const rl::State& s : env.states()) {
        if(env.is_end_state(s)) {
            continue;
        }
        EXPECT_NEAR(expected.value(s), value_function.value(s), max_error);
    }
}
