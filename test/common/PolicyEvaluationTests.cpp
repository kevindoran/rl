#include "PolicyEvaluationTests.h"
#include "suttonbarto/Exercise4_1.h"
#include "suttonbarto/Exercise5_1.h"

namespace {

rl::MappedEnvironment single_state_action_env(
        const std::string& state_name="State 1", const std::string& action_name="Action 1",
        double reward_value=1.0) {
    rl::MappedEnvironment env;
    const rl::State& state = env.add_state(state_name);
    const rl::Action& action = env.add_action(action_name);
    const rl::Reward& reward = env.add_reward(reward_value, "Reward 1");
    env.add_transition(rl::Transition(state, state, action, reward));
    env.build_distribution_tree();
    return env;
}

} // namespace

namespace rl {
namespace test {

GridWorldTest1::GridWorldTest1() {
    // Setup
    grid::Position top_left{0, 0};
    grid::Position bottom_left{HEIGHT - 1, 0};
    grid_world.set_start_state(grid_world.pos_to_state(top_left));
    grid_world.mark_as_end_state(grid_world.pos_to_state(bottom_left));
    grid_world.set_all_rewards_to(-1.0);
    p_down_up_policy = std::make_unique<DeterministicLambdaPolicy>(create_down_up_policy(grid_world));
}

void GridWorldTest1::check(StateBasedEvaluator& evaluator) const {
    // Test
    const ValueTable& value_function =
            evaluate(evaluator, grid_world, *p_down_up_policy);
    ASSERT_EQ(-4, value_function.value(grid_world.pos_to_state(grid::Position{0, 0})));
    ASSERT_EQ(-3, value_function.value(grid_world.pos_to_state(grid::Position{0, 1})));
    ASSERT_EQ(-2, value_function.value(grid_world.pos_to_state(grid::Position{0, 2})));
    ASSERT_EQ(-1, value_function.value(grid_world.pos_to_state(grid::Position{0, 3})));
    ASSERT_EQ(0, value_function.value(grid_world.pos_to_state(grid::Position{0, 4})));
}

void GridWorldTest1::check(ActionBasedEvaluator& evaluator) const {
    const ActionValueTable& value_function =
            evaluate(evaluator, grid_world, *p_down_up_policy);
    for(int h = 0; h < HEIGHT; h++) {
        for(grid::Direction d : grid::directions) {
            int direction_ordinal = static_cast<int>(d);
            double correct_value = expected_action_values[h][direction_ordinal];
            grid::Position pos{h, 0};
            const rl::Action& action = grid_world.dir_to_action(d);
            double value_to_test = value_function.value(grid_world.pos_to_state(pos), action);
            ASSERT_EQ(correct_value, value_to_test);
        }
    }
}

void SuttonBartoExercise4_1Test::check(StateBasedEvaluator& evaluator) const {
    // Setup
    using Ex4_1 = suttonbarto::Exercise4_1;
    Ex4_1 test_case;
    RandomPolicy policy;

    // Test
    const ValueTable& value_function = evaluate(evaluator, test_case.env(), policy);
    for (rl::ID state_id = 0; state_id < test_case.env().state_count(); state_id++) {
        ASSERT_NEAR(Ex4_1::expected_values[state_id],
                    value_function.value(test_case.env().state(state_id)),
                    ALLOWED_ERROR_FACTOR * std::abs(Ex4_1::expected_values[state_id]));
    }
}

void rl::test::ContinuousTaskTest::check(StateBasedEvaluator& evaluator) const {
    const int REWARD_VALUE = 5;
    MappedEnvironment env(single_state_action_env("State 1", "Action 1", REWARD_VALUE));
    FirstActionPolicy policy;
    for(int discount_rate_tenth = 1; discount_rate_tenth <= 9; discount_rate_tenth++) {
        double discount_rate = discount_rate_tenth / 10.0;
        evaluator.set_discount_rate(discount_rate);
        double denom = 1.0 - discount_rate;
        ASSERT_FALSE(denom == 0) << "The test implementation is broken if this fails.";
        double correct_value = REWARD_VALUE / denom;
        // We could be more exact here with out bounds.
        double bounds = ALLOWED_ERROR_FACTOR * correct_value;
        const State& the_only_state = *env.states().begin();
        const ValueTable& value_function = evaluate(evaluator, env, policy);
        ASSERT_NEAR(correct_value, value_function.value(the_only_state), bounds);
    }
}

void BrokenPolicyTest::check(StateBasedEvaluator& evaluator) const {
    // Setup
    MappedEnvironment env = single_state_action_env();
    // Set a discount rate so that the test doesn't go on forever in the case where the behaviour is
    // broken.
    evaluator.set_discount_rate(0.9);
    rl::test::NoActionPolicy no_action_policy;
    rl::test::ZeroWeightActionPolicy zero_weight_policy;

    // Test
    // 1. Exception expected for a policy with no action for a state.
    evaluator.initialize(env, no_action_policy);
    EXPECT_ANY_THROW(evaluator.run());

    // 2. Exception expected for a policy with an action of zero weight.
    evaluator.initialize(env, zero_weight_policy);
    EXPECT_ANY_THROW(evaluator.run());
}

void BlackjackSpecificCase::check(ActionBasedEvaluator& evaluator) const {
    // Setup
    using BlackjackEnv = suttonbarto::BlackjackEnvironment;
    BlackjackEnv env;
    const BlackjackEnv::BlackjackState start_state{15, false, 2};
    rl::DeterministicLambdaPolicy hit_then_stick(
            [&env, start_state](const rl::Environment&, const rl::State& state) -> const rl::Action& {
                if(start_state == env.blackjack_state(state)) {
                    return env.action(env.action_id(BlackjackEnv::BlackjackAction::HIT));
                } else {
                    return env.action(env.action_id(BlackjackEnv::BlackjackAction::STICK));
                }
            }
    );
    // FIXME: what should be done to get this threshold down?
    double allowed_error = 0.03;
    // From BlackjackEnvironmentF.test_specific_case_3, we know that the expected return from
    // (15, false, 2) with the hit-stick policy is:
    double expected_return = 0.267040 - 0.683266; // wins - losses (draws are: 0.049694)
    // Seed the generator to insure deterministic results.
    rl::util::random::reseed_generator(1);

    // Test
    const rl::ActionValueTable& value_fctn = rl::evaluate(evaluator, env, hit_then_stick);
    ASSERT_NEAR(
        expected_return,
        value_fctn.value(env.state(start_state), env.action(BlackjackEnv::BlackjackAction::HIT)),
        allowed_error);
}

} // namespace test
} // namespace rl