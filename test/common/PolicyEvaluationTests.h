#pragma once

#include <rl/GridWorld.h>
#include <rl/RandomPolicy.h>
#include "gtest/gtest.h"

#include "rl/Environment.h"
#include "rl/Policy.h"
#include "rl/DeterministicPolicy.h"
#include "ExamplePolicies.h"
#include "SuttonBartoExercises.h"


namespace rl {
namespace test {

class ActionBasedEvaluatorTestCase {
public:
    virtual void check(ActionBasedEvaluator& evaluator) const = 0;

    virtual ~ActionBasedEvaluatorTestCase() = default;
};

class StateBasedEvaluatorTestCase {
public:
    virtual void check(StateBasedEvaluator& evaluator) const = 0;

    virtual ~StateBasedEvaluatorTestCase() = default;
};

/*
 * The environment is a grid world with layout:
 *
 *  X
 *  X
 *  X
 *  X
 *  E
 *
 *  All actions produce a reward of -1.
 *
 * The policy always chooses down, unless down is not allowed, it which case it chooses up.
 */
class GridWorldTest1 : public StateBasedEvaluatorTestCase, ActionBasedEvaluatorTestCase {
public:
    GridWorldTest1();
    /**
     * For the state value function, the values should be:
     *
     * -4
     * -3
     * -2
     * -1
     *  0
     */
    void check(StateBasedEvaluator& evaluator) const override;

    /**
     * For the state-action value function, the values should be:
     * (columns are for each action)
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
    void check(ActionBasedEvaluator& evaluator) const override;

private:
    static const int HEIGHT = 5;
    static const int WIDTH = 1;
    static constexpr double expected_action_values[][grid::DIR_COUNT] =
        // In ordinal order.
        //  0     1    2    3
        // Right  Down Left Up
          { {-5,  -4,  -5,  -5},
            {-4,  -3,  -4,  -5},
            {-3,  -2,  -3,  -4},
            {-2,  -1,  -2,  -3},
            { 0,   0,   0,   0} };

private:
    GridWorld<HEIGHT, WIDTH> grid_world{GridWorldBoundsBehaviour::TRANSITION_TO_CURRENT};
    std::unique_ptr<DeterministicLambdaPolicy> p_down_up_policy;

};

class SuttonBartoExercise4_1Test : public StateBasedEvaluatorTestCase {
public:
    const double ALLOWED_ERROR_FACTOR = 0.02;

public:
    void check(StateBasedEvaluator& evaluator) const override;
};

// TODO: extend to include action based.
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
class ContinuousTaskTest : public StateBasedEvaluatorTestCase {
public:
    const double ALLOWED_ERROR_FACTOR = 0.01;
public:
    void check(StateBasedEvaluator& evaluator) const override;
};

class BrokenPolicyTest : public StateBasedEvaluatorTestCase {
public:
    void check(StateBasedEvaluator& evaluator) const override;
};

class BlackjackSpecificCase : public ActionBasedEvaluatorTestCase {
public:
    void check(ActionBasedEvaluator& evaluator) const override;
};


} // namespace rl
} // namespace test
