#include "gtest/gtest.h"

#include <unordered_set>
#include <rl/FirstVisitMCValuePredictor.h>
#include <ExamplePolicies.h>
#include <rl/ActionValuePolicyImprover.h>

#include "rl/GridWorld.h"
#include "grid/Grid.h"
#include "rl/Policy.h"
#include "common/SuttonBartoExercises.h"
#include "rl/DeterministicImprover.h"
#include "rl/RandomGridPolicy.h"

namespace {

// TODO: turn this into a gtest predicate.
void check_policy_action(const rl::Policy& policy, const rl::Environment& env,
                         const rl::State& in_state,
                         rl::test::TestEnvironment::OptimalActions optimal_actions) {
    rl::Policy::ActionDistribution action_dist = policy.possible_actions(env, in_state);
    auto weight_map = action_dist.weight_map();
    // End states shouldn't have an action.
    if(optimal_actions.empty()) {
        ASSERT_TRUE(weight_map.empty());
        // There is nothing left to test for end states.
        return;
    }
    // The policy must not contain more actions than the optimal policy set.
    ASSERT_GE(optimal_actions.size(), weight_map.size());
    // The policy's actions should be an optimal action.
    for(auto entry : weight_map) {
        const rl::Action& policy_action = *CHECK_NOTNULL(entry.first);
        ASSERT_TRUE(optimal_actions.count(policy_action.id()))
        << "Testing state: " << in_state.name() << " "
        << "correct: " << env.action(*optimal_actions.begin()).name()
        << ", actual: " << entry.first->name();
    }
}

void test_improver(rl::PolicyImprover& policy_improver,
                   const rl::test::TestEnvironment& test_case,
                   const rl::Policy& start_policy) {
    SCOPED_TRACE(test_case.name());
    // The evaluator.set_discount_rate() is wrapped by this if, as some evaluators throw an
    // exception due to lack of support for non-episodic tasks. TODO: worth revisiting.
    if(test_case.required_discount_rate() != 1.0) {
        policy_improver.policy_evaluator().set_discount_rate( test_case.required_discount_rate());
    }
    policy_improver.policy_evaluator().set_delta_threshold( test_case.required_delta_threshold());
    const rl::Environment& env = test_case.env();
    std::unique_ptr<rl::Policy> p_policy = policy_improver.improve(env, start_policy);
    ASSERT_TRUE(p_policy);
    for(int i = 0; i < env.state_count(); i++) {
        check_policy_action(*p_policy, env, env.state(i), test_case.optimal_actions(env.state(i)));
    }
}

} // namespace

TEST(PolicyImprovers, test_policy_iterator) {
    rl::DeterministicImprover improver;
    test_improver(improver, rl::test::Exercise4_1(), rl::test::RandomPolicy());
    test_improver(improver, rl::test::Exercise4_2(), rl::test::RandomPolicy());
    // FIXME: make pass and also allow for the assertions of failure.
    //test_improver(improver, rl::test::Exercise5_1(), rl::test::RandomPolicy());
}

TEST(PolicyImprovers, test_action_value_policy_iterator) {
    rl::ActionValuePolicyImprover improver;
    // FIXME: A Monte Carlo evaluator of deterministic policy on a deterministic environment
    //        has a high chance of encountering an infinite trial unless loop detection is
    //        implemented.
    //test_improver(improver, rl::test::Exercise4_1(), rl::test::FirstValidActionPolicy());
    test_improver(improver, rl::test::Exercise5_1(), rl::test::RandomPolicy());
}
