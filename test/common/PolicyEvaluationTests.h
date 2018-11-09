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
 *
 * With the down-up policy, the state values should be:
 *
 * -4
 * -3
 * -2
 * -1
 *  0
 */
class GridWorldTest1 : public StateBasedEvaluatorTestCase {
public:
    void check(StateBasedEvaluator& value_function) const override;
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

} // namespace rl
} // namespace test
