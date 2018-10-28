#include "gtest/gtest.h"

#include <unordered_set>

#include "rl/GridWorld.h"
#include "grid/Grid.h"
#include "rl/Policy.h"
#include "common/SuttonBartoExercises.h"
#include "rl/PolicyIterator.h"
#include "rl/RandomGridPolicy.h"

namespace {

void check_policy_action(const rl::Policy& policy, const rl::Environment& env,
                         const rl::State& in_state, const std::unordered_set<rl::ID>& optimal_actions) {
    rl::Policy::ActionDistribution action_dist = policy.possible_actions(env, in_state);
    auto weight_map = action_dist.weight_map();
    // End states shouldn't have an action.
    if(optimal_actions.empty()) {
        ASSERT_TRUE(weight_map.empty());
        // There is nothing left to test for end states.
        return;
    }
    // The policy must not contain more actions that the optimal policy set.
    ASSERT_GE(optimal_actions.size(), weight_map.size());
    // The policy's actions should be an optmal policy.
    for(auto entry : weight_map) {
        const rl::Action& policy_action = *CHECK_NOTNULL(entry.first);
        ASSERT_TRUE(optimal_actions.count(policy_action.id()))
        << "Testing state: " << in_state.name()
        << "correct: (" << env.action(*optimal_actions.begin()).name() << ")" << " vs "
        << "(" << entry.first->name() << ")";
    }
}

} // namespace
/**
 * This test calculates the optimum policy for the gridworld described in exercise 4.1 in
 * (Sutton & Barto, 2018). The initial policy is the random grid policy.
 *
 * The optimum policy is:
 * (actions available: U = up, D = down, L = left, R = right)
 *
 * X  L  L  DL
 * U  LU DL D
 * U  RU RD D
 * RU R  R  X
 *
 *
 */
TEST(PolicyIteration, sutton_barto_exercise_4_1) {
    auto grid_world = rl::test::Exercise4_1::create_grid_world();
    const rl::MappedEnvironment& env = grid_world.environment();
    rl::RandomGridPolicy random_policy(grid_world);
    rl::PolicyIterator policy_improver;
    std::unique_ptr<rl::Policy> p_policy = policy_improver.improve(env, random_policy);
    ASSERT_TRUE(p_policy);

    int R = grid::Direction::RIGHT;
    int D = grid::Direction::DOWN;
    int L = grid::Direction::LEFT;
    int U = grid::Direction::UP;
    std::unordered_set<int> optimal_actions[] = {
            {   }, {  L}, {  L}, {D,L},
            {  U}, {L,U}, {D,L}, {D,L},
            {  U}, {R,U}, {R,D}, {  D},
            {R,U}, {  R}, {  R}, {   }
    };

    for(int i = 0; i < grid_world.environment().state_count(); i++) {
        std::unordered_set<rl::ID> optimal_action_ids;
        std::transform(
            std::begin(optimal_actions[i]), std::end(optimal_actions[i]),
            std::inserter(optimal_action_ids, std::end(optimal_action_ids)),
            [&grid_world](int dir) {
                return grid_world.dir_to_action(grid::directions[dir]).id();
            }
            );
        check_policy_action(*p_policy, env, env.state(i), optimal_action_ids);
    }
}

class FirstValidActionPolicy : public rl::Policy {
public:
    const rl::Action &
    next_action(const rl::Environment &e, const rl::State &from_state) const override {
        for(const rl::Action& a : e.actions()) {
            if(!e.is_action_allowed(from_state, a)) {
                continue;
            }
            return a;
        }
        Ensures(false);
    }

    ActionDistribution
    possible_actions(const rl::Environment &e, const rl::State &from_state) const override {
        return ActionDistribution::single_action(next_action(e, from_state));
    }
};

/**
 * Tests that the description of the problem has been correctly represented.
 */
TEST(PolicyIteration, sutton_barto_exercise_4_2_setup) {
    // Setup
    rl::test::Exercise4_2::CarRentalEnvironment env;

    // Test cars_in_loc_X()
    const rl::State& one_in_each_loc = env.state(22);
    ASSERT_EQ(1, env.cars_in_loc_1(one_in_each_loc));
    ASSERT_EQ(1, env.cars_in_loc_2(one_in_each_loc));
    const rl::State& two_in_each_loc = env.state(44);
    ASSERT_EQ(2, env.cars_in_loc_1(two_in_each_loc));
    ASSERT_EQ(2, env.cars_in_loc_2(two_in_each_loc));

    // Test change in car count.
    const rl::Action& move_3_from_loc_1 = env.action(env.MAX_CAR_TRANSFERS + 3);
    const rl::Action& move_3_from_loc_2 = env.action(env.MAX_CAR_TRANSFERS - 3);
    ASSERT_EQ(-3, env.change_in_car_count(move_3_from_loc_1,
            rl::test::Exercise4_2::CarRentalEnvironment::Location::LOC1));
    ASSERT_EQ(3, env.change_in_car_count(move_3_from_loc_1,
            rl::test::Exercise4_2::CarRentalEnvironment::Location::LOC2));
    ASSERT_EQ(3, env.change_in_car_count(move_3_from_loc_2,
            rl::test::Exercise4_2::CarRentalEnvironment::Location::LOC1));
    ASSERT_EQ(-3, env.change_in_car_count(move_3_from_loc_2,
            rl::test::Exercise4_2::CarRentalEnvironment::Location::LOC2));

    // Test is_action_allowed.
    ASSERT_TRUE(env.is_action_allowed(env.state(3, 10), move_3_from_loc_1));
    ASSERT_TRUE(env.is_action_allowed(env.state(4, 10), move_3_from_loc_1));
    ASSERT_TRUE(env.is_action_allowed(env.state(4, 0), move_3_from_loc_1));
    ASSERT_FALSE(env.is_action_allowed(env.state(2, 10), move_3_from_loc_1));
    ASSERT_FALSE(env.is_action_allowed(env.state(2, 3), move_3_from_loc_1));
    ASSERT_FALSE(env.is_action_allowed(env.state(3, 19), move_3_from_loc_1));

    // Test the response calculations.
    // a) 0 cars -> 20 cars
    int prev_car_count = 0;
    int new_car_count = env.MAX_CAR_COUNT;
    double rental_prob = 1.0; // Any rental count matches.
    double return_prob = gsl_cdf_poisson_Q(env.MAX_CAR_COUNT-1, env.LOC1_RETURN_MEAN);
    double correct_prob = rental_prob * return_prob;
    auto response_part = env.possibilities(prev_car_count, new_car_count, env.LOC1_RENTAL_MEAN, env.LOC1_RETURN_MEAN);
    ASSERT_EQ(correct_prob, response_part.probability);
    ASSERT_EQ(0, response_part.revenue);

    // b) 20 cars -> 0 cars
    prev_car_count = env.MAX_CAR_COUNT;
    new_car_count = 0;
    rental_prob = gsl_cdf_poisson_Q(env.MAX_CAR_COUNT-1, env.LOC1_RENTAL_MEAN);
    return_prob = gsl_ran_poisson_pdf(0, env.LOC1_RETURN_MEAN);
    correct_prob = rental_prob * return_prob;
    response_part = env.possibilities(prev_car_count, new_car_count, env.LOC1_RENTAL_MEAN, env.LOC1_RETURN_MEAN);
    ASSERT_EQ(correct_prob, response_part.probability);
    ASSERT_EQ(env.INCOME_PER_RENTAL * prev_car_count, response_part.revenue);

    // c) 1 cars -> 2 cars
    prev_car_count = 1;
    new_car_count = 2;
    // There are two cases:
    // 1. Rent no cars.
    double rental_prob_1 = gsl_ran_poisson_pdf(0, env.LOC1_RENTAL_MEAN);
    double return_prob_1 = gsl_ran_poisson_pdf(1, env.LOC1_RETURN_MEAN);
    double correct_prob_1 = rental_prob_1 * return_prob_1;
    double income_1 = 0;
    // 2. Rent 1 car.
    double rental_prob_2 = gsl_cdf_poisson_Q(0, env.LOC1_RENTAL_MEAN); // ie (1 - rental_prob_1).
    double return_prob_2 = gsl_ran_poisson_pdf(2, env.LOC1_RETURN_MEAN);
    double correct_prob_2 = rental_prob_2 * return_prob_2;
    int income_2 = 1* env.INCOME_PER_RENTAL;
    correct_prob = correct_prob_1 + correct_prob_2;
    double expected_income = income_2 * (correct_prob_2 / correct_prob);
    response_part = env.possibilities(prev_car_count, new_car_count, env.LOC1_RENTAL_MEAN, env.LOC1_RETURN_MEAN);
    ASSERT_DOUBLE_EQ(correct_prob, response_part.probability);
    ASSERT_DOUBLE_EQ(expected_income, response_part.revenue);
}


// Disabled temporarily:
//   * The results are slight different from those presented by Sutton & Barto (5 tiles differ).
//   * The test takes about 1 minute to run, which is disruptive to development at the moment.
// TODO: figure out why there is a discrepancy with the results.
// TODO: improve the performance.
TEST(PolicyIteration, DISABLED_sutton_barto_exercise_4_2) {
    // Setup
    rl::test::Exercise4_2::CarRentalEnvironment env;
    rl::PolicyIterator policy_improver;
    policy_improver.policy_evaluator().set_discount_rate(0.9);
    auto next_state_fctn = [&env](const rl::Environment& e, const rl::State& from_state) -> const rl::Action& {
        const rl::Action& a = e.action(env.action_id(0));
        return a;
    };
    rl::DeterministicLambdaPolicy start_policy(next_state_fctn);
    // Copying the values from the book.
    int optimal_policy[env.MAX_CAR_COUNT + 1][env.MAX_CAR_COUNT + 1] =
           // 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
           { {0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -2, -2, -2, -3, -3, -3, -3, -3, -4, -4, -4}, //  0
             {0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -2, -2, -2, -2, -2, -3, -3, -3, -3}, //  1
             {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2}, //  2
             {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -2}, //  3
             {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1}, //  4
             {1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, //  5
             {2,  2,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, //  6
             {3,  2,  2,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, //  7
             {3,  3,  2,  2,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, //  8
             {4,  3,  3,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, //  9
             {4,  4,  3,  3,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 10
             {5,  4,  4,  3,  2,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 11
             {5,  5,  4,  3,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 12
             {5,  5,  4,  3,  3,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 13
             {5,  5,  4,  4,  3,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 14
             {5,  5,  5,  4,  3,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 15
             {5,  5,  5,  4,  3,  2,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 16
             {5,  5,  5,  4,  3,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 17
             {5,  5,  5,  4,  3,  3,  2,  2,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 18
             {5,  5,  5,  4,  4,  3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0}, // 19
             {5,  5,  5,  5,  4,  4,  3,  3,  3,  3,  2,  2,  2,  2,  2,  1,  1,  1,  0,  0,  0}  // 20
           };

    // Test
    std::unique_ptr<rl::Policy> p_policy = policy_improver.improve(env, start_policy);
    ASSERT_TRUE(p_policy);
    for(int l1 = 0; l1 <= env.MAX_CAR_COUNT; l1++) {
        for(int l2 = 0; l2 <= env.MAX_CAR_COUNT; l2++) {
            const rl::State& state = env.state(l1, l2);
            int optimal_cars_moved = optimal_policy[l1][l2];
            std::unordered_set<rl::ID> optimal_action_ids {env.action_id(optimal_cars_moved)};
            check_policy_action(*p_policy, env, state, optimal_action_ids);
        }
    }
}