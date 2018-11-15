#include "gtest/gtest.h"
#include <unordered_set>
#include <suttonbarto/Example6_5.h>

#include "common/ExamplePolicies.h"
#include "rl/FirstVisitMCValuePredictor.h"
#include "rl/ActionValuePolicyImprover.h"
#include "rl/MCEvaluator3.h"
#include "rl/SarsaImprover.h"

#include "rl/GridWorld.h"
#include "grid/Grid.h"
#include "rl/Policy.h"
#include "common/TestEnvironment.h"
#include "common/suttonbarto/Exercise4_1.h"
#include "common/suttonbarto/Exercise4_2.h"
#include "common/suttonbarto/Exercise5_1.h"
#include "rl/DeterministicImprover.h"
#include "rl/RandomPolicy.h"
#include "rl/Trial.h"

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
    // Seed the generator to insure deterministic results.
    rl::util::random::reseed_generator(1);
    // The evaluator.set_discount_rate() is wrapped by this if, as some evaluators throw an
    // exception due to lack of support for non-episodic tasks. TODO: worth revisiting.
    if(test_case.required_discount_rate() != 1.0) {
        policy_improver.set_discount_rate( test_case.required_discount_rate());
    }
    policy_improver.set_delta_threshold(test_case.required_delta_threshold());
    const rl::Environment& env = test_case.env();
    std::unique_ptr<rl::Policy> p_policy = policy_improver.improve(env, start_policy);
    ASSERT_TRUE(p_policy);
    for(int i = 0; i < env.state_count(); i++) {
        check_policy_action(*p_policy, env, env.state(i), test_case.optimal_actions(env.state(i)));
    }
}

} // namespace

TEST(PolicyImprovers, policy_iterator_LONG_RUNNING) {
    rl::DeterministicImprover improver;
    test_improver(improver, rl::test::suttonbarto::Exercise4_1(), rl::RandomPolicy());
    test_improver(improver, rl::test::suttonbarto::Exercise4_2(), rl::RandomPolicy());
    // FIXME: make pass and also allow for the assertions of failure.
    //test_improver(improver, rl::test::Exercise5_1(), rl::test::RandomPolicy());
}

TEST(PolicyImprovers, action_value_policy_iterator_LONG_RUNNING) {
    rl::ActionValuePolicyImprover improver;
    // FIXME: A Monte Carlo evaluator of deterministic policy on a deterministic environment
    //        has a high chance of encountering an infinite trial unless loop detection is
    //        implemented.
    //test_improver(improver, rl::test::Exercise4_1(), rl::test::FirstValidActionPolicy());
    test_improver(improver, rl::test::suttonbarto::Exercise5_1(), rl::RandomPolicy());
}

TEST(PolicyImprovers, action_value_iterator_with_MCEvalutar3_LONG_RUNNING) {
    // Setup
    rl::ActionValuePolicyImprover improver;
    rl::MCEvaluator3 evaluator;
    improver.set_policy_evaluator(evaluator);
    // Test
    test_improver(improver, rl::test::suttonbarto::Exercise5_1(), rl::RandomPolicy());
}

TEST(PolicyImprovers, sarsa_improver_LONG_RUNNING) {
    // Setup
    rl::SarsaImprover improver;
    // Test
    test_improver(improver, rl::test::suttonbarto::Exercise5_1(), rl::RandomPolicy());
}

TEST(PolicyImprovers, sarsa_example_6_5) {
    // Setup
    rl::SarsaImprover sarsa;
    sarsa.set_delta_threshold(0.001);
    //rl::DeterministicImprover sarsa;
    using Ex6_5 = rl::test::suttonbarto::Example6_5;
    Ex6_5::WindyGridWorld windy_grid_world;
    rl::util::random::reseed_generator(1);
    rl::RandomPolicy start_policy;

    // Test
    // policy_improver.set_delta_threshold(test_case.required_delta_threshold());
    std::unique_ptr<rl::Policy> p_policy = sarsa.improve(windy_grid_world, start_policy);
    ASSERT_TRUE(p_policy);
    rl::Trace trace = rl::run_trial(windy_grid_world, *p_policy);
    ASSERT_EQ(trace.size(), Ex6_5::OPTIMAL_ROUTE.size());
    for(int i = 0; i < static_cast<int>(Ex6_5::OPTIMAL_ROUTE.size()); i++) {
        const rl::State& expected = windy_grid_world.pos_to_state(Ex6_5::OPTIMAL_ROUTE.at(i));
        ASSERT_EQ(expected, trace.at(i).state)
            << "The calculated policy should produce the optimal route when used.";
    }
}