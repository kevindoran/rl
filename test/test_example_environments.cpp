#include "gtest/gtest.h"

#include <numeric>

#include "common/SuttonBartoExercises.h"

/**
 * Tests that the description of the Jack's Car Garage problem has been correctly represented.
 */
 // TODO: split into separate tests.
TEST(ExampleEnvironments, sutton_barto_exercise_4_2_jacks_garage) {
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

using BlackjackEnv = rl::test::Exercise5_1::BlackjackEnvironment;
class BlackjackEnvironmentF : public ::testing::Test {
protected:
    BlackjackEnv env;
};

TEST_F(BlackjackEnvironmentF, test_action_mapping) {
    ASSERT_EQ(2, env.action_count());
    ASSERT_EQ(0, env.action_id(env.blackjack_action(env.action(0))));
    ASSERT_EQ(1, env.action_id(env.blackjack_action(env.action(1))));
    ASSERT_EQ(BlackjackEnv::BlackjackAction::HIT,
              env.blackjack_action(env.action(env.action_id(BlackjackEnv::BlackjackAction::HIT))));
    ASSERT_EQ(BlackjackEnv::BlackjackAction::STICK,
              env.blackjack_action(env.action(env.action_id(BlackjackEnv::BlackjackAction::STICK))));
}

TEST_F(BlackjackEnvironmentF, test_state_mapping) {
    for(int dealer_card = BlackjackEnv::ACE; dealer_card <= BlackjackEnv::TEN; dealer_card++) {
        for(int player_sum = 12; player_sum <= BlackjackEnv::MAX_SUM; player_sum++) {
            for(bool usable_ace : {true, false}) {
                BlackjackEnv::BlackjackState bj_state{player_sum, usable_ace, dealer_card};
                BlackjackEnv::BlackjackState matching_bj_state =
                        env.blackjack_state(env.state(env.state_id(bj_state)));
                ASSERT_EQ(bj_state, matching_bj_state);
            }
        }
    }
    for(const rl::State& state : env.states()) {
        // End states don't have a matching BlackjackState.
        if(env.is_end_state(state)) {
            continue;
        }
        ASSERT_EQ(state, env.state(env.state_id(env.blackjack_state(state))));
    }
}

// It's hard to thoroughly test the transition_list() method.
// The few tests below just test some of the obvious cases.
/**
 *  1. Player hits when going bust is impossible.
 */
TEST_F(BlackjackEnvironmentF, test_transition_list_hit) {
    // Setup.
    const int player_sum = 12;
    const bool usable_ace = true;
    const int dealer_card = BlackjackEnv::ACE;
    BlackjackEnv::BlackjackState blackjack_state{player_sum, usable_ace, dealer_card};

    // Test.
    rl::ResponseDistribution res = env.transition_list(
            env.state(env.state_id(blackjack_state)),
            env.action(env.action_id(BlackjackEnv::BlackjackAction::HIT)));
    // There should be 10 transitions, with 10% chance each.
    ASSERT_EQ(10, res.responses().size());
    ASSERT_DOUBLE_EQ(1.0, res.total_weight());
    for(const rl::Response& r : res.responses()) {
        ASSERT_DOUBLE_EQ(0.1, r.prob_weight);
    }
    // The 10 transitions are known and are as follows.
    for(int card = BlackjackEnv::ACE; card <= BlackjackEnv::TEN; card++) {
        int expected_player_sum = player_sum + env.card_value(card);
        bool expected_usable_ace = true;
        if(card == BlackjackEnv::ACE) {
            // 12 + 11 would be > 21, to use as a 1.
            expected_player_sum = env.revert_ace(expected_player_sum);
        } else if(card == BlackjackEnv::TEN) {
            expected_player_sum = env.revert_ace(expected_player_sum);
            expected_usable_ace = false;
        }
        BlackjackEnv::BlackjackState expected_state{
            expected_player_sum, expected_usable_ace, dealer_card};
        auto finder = std::find_if(std::begin(res.responses()), std::end(res.responses()),
               [&expected_state, this](const rl::Response& res) {
                   return env.state(env.state_id(expected_state)) == res.next_state;
               }
           );
        ASSERT_TRUE(finder != std::end(res.responses()))
            << "Couldn't find the response for card_id: " << card;
    }
}

/**
 * 2. Player has 21 and sticks.
 */
TEST_F(BlackjackEnvironmentF, test_transition_list_21_stick) {
    // Setup.
    const int player_sum = 21;
    const bool usable_ace = true;
    const int dealer_card = BlackjackEnv::TEN;
    BlackjackEnv::BlackjackState blackjack_state{player_sum, usable_ace, dealer_card};

    // Calculate the chance of drawing via DP.
    // To save tedious index management, just make the array size 22. We will only use from 11-21.
    double draw_chances[21 + 1] = {0};
    draw_chances[21] = 1.0;
    for(int from = 16; from >= 10; from--) {
        double probability = 0.0;
        for(int card_value = 1; card_value <= 11; card_value++) {
            // 10 Will only choose 11 as the ace value.
            if(from == 10 and card_value == 1) {
                continue;
            }
            int total = from + card_value;
            if(total > 21) {
                continue;
            } else {
                probability += 0.1 * draw_chances[total];
            }
        }
        draw_chances[from] = probability;
    }
    double draw_probability = draw_chances[10];
    double win_probability = 1.0 - draw_probability;

    // note: at first, the DP solution disagreed with the observered result, so I calculated it
    // by-hand to check which was correct. The discrepancy was tracked down to a bug in
    // BlackjackEnvironment. Leaving both here for now.
    //
    // This whole process highlights a major drawback of the deterministic (non-Monte Carlo)
    // approaches: it is tedious and error prone to describe the environment dynamics accurately.
    //
    // Case      Combination   Probability    Example
    // 1 card:     5,5          1x(1/10)      10, 21
    // 2 cards:    5,4          5x(1/10)^2    10, 12, 21
    // 3 cards:    5,3         10x(1/10)^3    10, 12, 14, 21
    // 4 cards:    5,2         10x(1/10)^4    10, 12, 14, 16, 21
    // 5 cards:    5,1          5x(1/10)^5    10, 12, 14, 15, 16, 21
    // 6 cards:    5,1          1x(1/10)^6    10, 12, 13, 14, 15, 16, 21
    //
    // Total: 0.161051
    ASSERT_EQ(0.161051, draw_probability);

    // Test.
    rl::ResponseDistribution res = env.transition_list(
            env.state(env.state_id(blackjack_state)),
            env.action(env.action_id(BlackjackEnv::BlackjackAction::STICK)));
    ASSERT_EQ(2, res.responses().size());
    ASSERT_DOUBLE_EQ(1.0, res.total_weight());
    for (const rl::Response& r : res.responses()) {
        if (r.next_state == env.win_state()) {
            ASSERT_DOUBLE_EQ(win_probability, r.prob_weight);
        } else if (r.next_state == env.draw_state()) {
            ASSERT_DOUBLE_EQ(draw_probability, r.prob_weight);
        } else {
            ASSERT_TRUE(false) << "There should be no other transitions.";
        }
    }
}
