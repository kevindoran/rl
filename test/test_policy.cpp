#include "gtest/gtest.h"

#include "rl/Policy.h"

/**
 * Tests that 0 is returned by an action distribution when querying for an action that isn't listed
 * in the action distribution. No exceptions should be thrown.
 *
 * At one point, the behaviour was to throw an exception in this case. This became a problem when
 * implementing importance sampling off-policy evaluation where it is often that a policy doesn't
 * have any weight assigned to a action that is taken by the behaviour policy.
 */
TEST(ActionDistribution, query_for_zero_weight_action) {
    // Setup
    rl::Policy::ActionDistribution action_dist;
    rl::Action a0(0, "Action 0");
    rl::Action a1(1, "Action 1");
    rl::Weight a0_weight = 1.0;
    action_dist.add_action(a0, a0_weight);

    // Test
    ASSERT_NO_THROW(action_dist.weight(a1));
}