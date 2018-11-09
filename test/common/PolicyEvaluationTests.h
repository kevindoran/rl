#pragma once

#include <rl/GridWorld.h>
#include "gtest/gtest.h"

#include "rl/Environment.h"
#include "rl/Policy.h"
#include "rl/DeterministicPolicy.h"
#include "ExamplePolicies.h"

namespace rl {
namespace test {

class ActionBasedEvaluatorTestCase {
public:
    virtual const rl::Environment& env() const = 0;
    virtual const rl::Policy& policy() const = 0;
    virtual ::testing::AssertionResult check(const rl::ActionValueFunction& value_function) const = 0;
    virtual std::string name() const = 0;

    virtual ~ActionBasedEvaluatorTestCase() = default;

};

class StateBasedEvaluatorTestCase {
public:
    virtual const rl::Environment& env() const = 0;
    virtual const rl::Policy& policy() const = 0;
    virtual void check(const rl::ValueFunction& value_function) const = 0;
    virtual std::string name() const = 0;

    virtual ~StateBasedEvaluatorTestCase() = default;
};


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

    std::string name() const override {
        return "GridWorldTest1 (down-up policy on a 1-D grid)";
    }

private:
    static const int HEIGHT = 5;
    static const int WIDTH = 1;
    rl::GridWorld<HEIGHT, WIDTH> grid_world{rl::GridWorldBoundsBehaviour::TRANSITION_TO_CURRENT};
    std::unique_ptr<rl::DeterministicLambdaPolicy> p_down_up_policy;
};

// We could use a template for these two functions, but they are so simple that there isn't much benefit.
void test_evaluator(rl::StateBasedEvaluator& evaluator,
                    const StateBasedEvaluatorTestCase& test_case);

void test_evaluator(rl::ActionBasedEvaluator& evaluator,
                    const ActionBasedEvaluatorTestCase& test_case);

} // namespace rl
} // namespace test
