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
    virtual const rl::Environment& env() const = 0;
    virtual const rl::Policy& policy() const = 0;
    virtual void check(const rl::ActionValueFunction& value_function) const = 0;

    virtual ~ActionBasedEvaluatorTestCase() = default;

};

class StateBasedEvaluatorTestCase {
public:
    virtual const rl::Environment& env() const = 0;
    virtual const rl::Policy& policy() const = 0;
    virtual void check(const rl::ValueFunction& value_function) const = 0;

    virtual ~StateBasedEvaluatorTestCase() = default;
};

// We could use a template for these two functions, but they are so simple that there isn't much benefit.
void test_evaluator(rl::StateBasedEvaluator& evaluator,
                    const StateBasedEvaluatorTestCase& test_case);

void test_evaluator(rl::ActionBasedEvaluator& evaluator,
                    const ActionBasedEvaluatorTestCase& test_case);

class GridWorldTest1 : public StateBasedEvaluatorTestCase {
public:
    GridWorldTest1();

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
     */
    const rl::Environment& env() const override {
        return grid_world.environment();
    }

    /**
     * The policy always chooses down, unless down is not allowed, it which case it chooses up.
     */
    const rl::Policy& policy() const override {
        return *p_down_up_policy;
    }

    /**
     * With the down-up policy, the state values should be:
     *
     * -4
     * -3
     * -2
     * -1
     *  0
     */
    void check(const rl::ValueFunction& value_function) const override;

private:
    static const int HEIGHT = 5;
    static const int WIDTH = 1;
    rl::GridWorld<HEIGHT, WIDTH> grid_world{rl::GridWorldBoundsBehaviour::TRANSITION_TO_CURRENT};
    std::unique_ptr<rl::DeterministicLambdaPolicy> p_down_up_policy;
};

// TODO: decide how to disabiguate these classes from the Sutton & Barto environments.
class SuttonBartoExercise4_1 : public StateBasedEvaluatorTestCase {
public:
    const double ALLOWED_ERROR_FACTOR = 0.02;

public:
    const Environment& env() const override {
        return test_case.env();
    }

    const Policy& policy() const override {
        return policy_;
    }

    void check(const rl::ValueFunction& value_function) const override;

private:
    using Ex4_1 = rl::test::Exercise4_1;
    Ex4_1 test_case;
    rl::RandomPolicy policy_;
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
    explicit ContinuousTaskTest(double discount_rate);

    const Environment& env() const override {
        return env_;
    }

    const Policy& policy() const override {
        return policy_;
    }

    void check(const ValueFunction& value_function) const override;

private:
    double discount_rate;
    static const int REWARD_VALUE = 5;
    MappedEnvironment env_;
    FirstActionPolicy policy_;
};

} // namespace rl
} // namespace test
