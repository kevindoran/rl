#include "gtest/gtest.h"

#include <unordered_set>
#include <rl/FirstVisitMCValuePrediction.h>
#include <ExamplePolicies.h>

#include "rl/GridWorld.h"
#include "grid/Grid.h"
#include "rl/Policy.h"
#include "common/SuttonBartoExercises.h"
#include "rl/PolicyIterator.h"
#include "rl/RandomGridPolicy.h"

namespace {

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
        << "correct: (" << env.action(*optimal_actions.begin()).name() << ")" << " vs "
        << "(" << entry.first->name() << ")";
    }
}


using TestEnvironments = std::vector<std::unique_ptr<rl::test::TestEnvironment>>;
TestEnvironments create_test_case_list() {
    // Sadly, the following is not possible. (https://stackoverflow.com/questions/9618268/initializing-container-of-unique-ptrs-from-initializer-list-fails-with-gcc-4-7)
    /*
    return TestEnvironments{
            std::make_unique<rl::test::Exercise4_1>(),
            std::make_unique<rl::test::Exercise4_2>(),
            std::make_unique<rl::test::Exercise5_1>()
        };
    */
    TestEnvironments environments;
    environments.emplace_back(std::make_unique<rl::test::Exercise4_1>());
    environments.emplace_back(std::make_unique<rl::test::Exercise4_2>());
    //environments.emplace_back(std::make_unique<rl::test::Exercise5_1>());
    return environments;
}

using StartingPolicies = std::vector<std::unique_ptr<rl::Policy>>;
StartingPolicies create_starting_policies() {
    StartingPolicies policies;
    // FIXME: PolicyIterator fails using FirstValidActionPolicy as a start policy.
    // policies.emplace_back(std::make_unique<rl::test::FirstValidActionPolicy>());
    policies.emplace_back(std::make_unique<rl::test::RandomPolicy>());
    return policies;
}

void test_improver(rl::PolicyImprovement& policy_improver,
                   const rl::test::TestEnvironment& test_case,
                   const rl::Policy& start_policy) {
    const rl::Environment& env = test_case.env();
    std::unique_ptr<rl::Policy> p_policy = policy_improver.improve(test_case.env(), start_policy);
    ASSERT_TRUE(p_policy);
    for(int i = 0; i < env.state_count(); i++) {
        check_policy_action(*p_policy, env, env.state(i), test_case.optimal_actions(env.state(i)));
    }
}

void test_improver(rl::PolicyIterator& policy_improver,
                   const TestEnvironments& test_cases,
                   const StartingPolicies& starting_policies) {
    for(const auto& p_test_case : test_cases) {
        SCOPED_TRACE(p_test_case->name());
        for(int i = 0; i < static_cast<int>(starting_policies.size()); i++) {
            SCOPED_TRACE(std::to_string(i));
            const auto& p_starting_policy = starting_policies[i];
            // TODO: should all policies have a name?
            // TODO: we may need to make all the arguments factory methods that construct each of
            //       the objects. As it currently is, there is the potential for interference.
            policy_improver.policy_evaluator().set_discount_rate(
                    p_test_case->required_discount_rate());
            policy_improver.policy_evaluator().set_delta_threshold(
                    p_test_case->required_delta_threshold());
            test_improver(policy_improver, *p_test_case, *p_starting_policy);
        }
    }
}

} // namespace

TEST(PolicyImprover, test_policy_iterator) {
    rl::PolicyIterator improver;
    test_improver(improver, create_test_case_list(), create_starting_policies());
}
