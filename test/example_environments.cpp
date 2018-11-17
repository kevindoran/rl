#include "gtest/gtest.h"

#include <numeric>
#include <suttonbarto/Example6_5.h>

#include "rl/DeterministicPolicy.h"
#include "rl/Trial.h"
#include "common/suttonbarto/Exercise4_2.h"
#include "common/suttonbarto/Exercise5_1.h"

namespace sb = rl::test::suttonbarto;
/**
 * Tests that the description of the Jack's Car Garage problem has been correctly represented.
 */
 // TODO: split into separate tests.
TEST(ExampleEnvironments, sutton_barto_exercise_4_2_jacks_garage) {
    // Setup
    sb::Exercise4_2::CarRentalEnvironment env;

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
                                          sb::Exercise4_2::CarRentalEnvironment::Location::LOC1));
    ASSERT_EQ(3, env.change_in_car_count(move_3_from_loc_1,
                                          sb::Exercise4_2::CarRentalEnvironment::Location::LOC2));
    ASSERT_EQ(3, env.change_in_car_count(move_3_from_loc_2,
                                          sb::Exercise4_2::CarRentalEnvironment::Location::LOC1));
    ASSERT_EQ(-3, env.change_in_car_count(move_3_from_loc_2,
                                          sb::Exercise4_2::CarRentalEnvironment::Location::LOC2));

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

using BlackjackEnv = sb::BlackjackEnvironment;
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
    // There should be 10 transitions, with 1/13 chance each (or 4/13 for card 10).
    ASSERT_EQ(static_cast<std::size_t>(10), res.responses().size());
    ASSERT_DOUBLE_EQ(1.0, res.total_weight());
    for(const rl::Response& r : res.responses()) {
        int was_ten = (env.blackjack_state(r.next_state).player_sum) == player_sum;
        if(was_ten) {
            ASSERT_DOUBLE_EQ(4.0/13.0, r.prob_weight);
        } else {
            ASSERT_DOUBLE_EQ(1.0/13.0, r.prob_weight);
        }
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
            double card_chance = (card_value == 10) ? 4.0/13.0 : 1.0/13.0;
            // 10 Will only choose 11 as the ace value.
            if(from == 10 and card_value == 1) {
                continue;
            }
            int total = from + card_value;
            if(total > 21) {
                continue;
            } else {
                probability += card_chance * draw_chances[total];
            }
        }
        draw_chances[from] = probability;
    }
    // These are our expected results:
    double draw_probability = draw_chances[10];
    double win_probability = 1.0 - draw_probability;

    // Test.
    rl::ResponseDistribution res = env.transition_list(
            env.state(env.state_id(blackjack_state)),
            env.action(env.action_id(BlackjackEnv::BlackjackAction::STICK)));
    ASSERT_EQ(static_cast<std::size_t>(2), res.responses().size());
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

struct WinDrawLoss {
    double wins = 0;
    double draws = 0;
    double losses = 0;
};

/**
 * Tests that the Blackjack simulation behaves as expected in terms of expected outcomes.
 *
 * \param env a blackjack environment.
 * \param from_state the state from which
 * \param expected
 */
void test_specific_case(BlackjackEnv& env, BlackjackEnv::BlackjackState from_state, const
                        rl::Policy& policy, WinDrawLoss expected) {
    // Setup
    int loops = 500000;
    WinDrawLoss observed{};
    double confidence_required = 0.95;
    // The input is taken as ratios. Transform these into expected counts.
    expected.wins *= loops;
    expected.draws *= loops;
    expected.losses *= loops;
    rl::util::random::reseed_generator(1);

    // Test
    for(int i = 0; i < loops; i++) {
        rl::Trial trial(env, env.state(env.state_id(from_state)));
        rl::Trace trace;
        const rl::Action* action = nullptr;
        while(!trial.is_finished()) {
            trace.emplace_back(rl::TimeStep{trial.current_state(), action, 0});
            action = &policy.next_action(env, trial.current_state());
            trial.execute_action(*action);
        }
        if(trial.current_state() == env.win_state()) {
            observed.wins++;
        } else if(trial.current_state() == env.draw_state()) {
            observed.draws++;
        } else {
            ASSERT_EQ(trial.current_state(), env.loss_state());
            observed.losses++;
        }
    }
    // Chi-squared test
    // Following steps outlined at: https://stattrek.com/chi-square-test/goodness-of-fit.aspx
    // Using the method: X^2 = ( (O-E)^2 / E )
    double X2 = 0;;
    int degrees_of_freedom = -1;
    if(expected.wins) {
        X2 += std::pow(observed.wins   - expected.wins,   2) / expected.wins;
        degrees_of_freedom++;
    } else {
        ASSERT_EQ(0, observed.wins);
    }
    if(expected.draws) {
        X2 += std::pow(observed.draws  - expected.draws,  2) / expected.draws;
        degrees_of_freedom++;
    } else {
        ASSERT_EQ(0, observed.draws);
    }
    if(expected.losses) {
        X2 += std::pow(observed.losses - expected.losses, 2) / expected.losses;
        degrees_of_freedom++;
    } else {
        ASSERT_EQ(0, observed.losses);
    }
    const double p_value = 1 - gsl_cdf_chisq_P(X2, degrees_of_freedom);
    const double cut_off = 1 - confidence_required;
    ASSERT_GT(p_value, cut_off);
}

/**
 * Tests the win/draw/loss ratio for the state-action pair:
 *
 *     (player_sum=17, usable_ace=false, dealer_card = ACE), action = STICK
 */
TEST_F(BlackjackEnvironmentF, test_specific_case_1) {
    // Setup
    const BlackjackEnv::BlackjackState start_state{17, false, BlackjackEnv::ACE};
    const WinDrawLoss expected_ratio {
        0.115333, /* wins */
        0.130662, /* draws */
        0.754005  /* losses */
    };
    rl::DeterministicLambdaPolicy stick_policy(
            [this](const rl::Environment&, const rl::State&) -> const rl::Action& {
                return env.action(env.action_id(BlackjackEnv::BlackjackAction::STICK));
            }
        );

    // Test
    test_specific_case(env, start_state, stick_policy, expected_ratio);
}

/**
 * Tests the win/draw/loss ratio for the state-action pair:
 *
 *     (player_sum=15, usable_ace=false, dealer_card = 2), action = STICK
 */
TEST_F(BlackjackEnvironmentF, test_specific_case_2) {
    // Setup
    const BlackjackEnv::BlackjackState start_state{15, false, 2};
    const WinDrawLoss expected_ratio {
            0.353984, /* wins */
            0.0,      /* draws */
            0.646016  /* losses */
    };
    rl::DeterministicLambdaPolicy stick_policy(
            [this](const rl::Environment&, const rl::State&) -> const rl::Action& {
                return env.action(env.action_id(BlackjackEnv::BlackjackAction::STICK));
            }
    );

    // Test
    test_specific_case(env, start_state, stick_policy, expected_ratio);
}

/**
 * Tests the win/draw/loss ratio for the policy: (hit, stick) from the state (15, false, 2).
 *
 * The start state:
 *     (player_sum=15, usable_ace=false, dealer_card = 2)
 * The policy: hit on the first turn, stick on the second.
 *
 * The ratios were calculated manually.
 */
TEST_F(BlackjackEnvironmentF, test_specific_case_3) {
    // Setup
    const BlackjackEnv::BlackjackState start_state{15, false, 2};
    const WinDrawLoss expected_ratio {
            0.267040, /* wins */
            0.049694, /* draws */
            0.683266  /* losses */
    };
    rl::DeterministicLambdaPolicy hit_then_stick(
            [this, start_state](const rl::Environment&, const rl::State& state) -> const rl::Action& {
                if(start_state == env.blackjack_state(state)) {
                    return env.action(env.action_id(BlackjackEnv::BlackjackAction::HIT));
                } else {
                    return env.action(env.action_id(BlackjackEnv::BlackjackAction::STICK));
                }
            }
    );

    // Test
    test_specific_case(env, start_state, hit_then_stick, expected_ratio);
}
/**
 * This test is used to generate dealer sum probabilities for creating the expected ratios in above
 * tests.
 */
/*
TEST_F(BlackjackEnvironmentF, test_simulate_dealer) {
    const int loops = 1000000;
    std::map<int, int> sum_counts;
    for(int i = 0; i < loops; i++) {
        int sum = env.simulate_dealer_turn(1);
        ++sum_counts[sum];
    }
    std::cout
              << "16: " << sum_counts[16] / (double) loops
              << ", 17: " << sum_counts[17] / (double) loops
              << ", 18: " << sum_counts[18] / (double)loops
              << ", 19: " << sum_counts[19] / (double)loops
              << ", 20: " << sum_counts[20] / (double)loops
              << ", 21: " << sum_counts[21] / (double)loops
              << ", 22: " << sum_counts[22] / (double)loops
              << ", 23: " << sum_counts[23] / (double)loops
              << ", 24: " << sum_counts[24] / (double)loops
              << ", 25: " << sum_counts[25] / (double)loops
              << ", 26: " << sum_counts[26] / (double)loops
              << std::endl;
    // Dealer card = 2:
    // 17: 0.13993, 18: 0.134533, 19: 0.130004, 20: 0.123439, 21: 0.118489
}*/

/**
 * Tests the wind effect on WindyGridWorld.
 *
 * Tests that:
 *   1. next_state() acts like normal GridWorld for tiles that don't have wind.
 *   2. next_state() applies wind _before_ carrying out the action.
 *   3. wind doesn't blow off the grid.
 */
TEST(WindyGridWorld, next_state) {
    // Setup
    sb::Example6_5::WindyGridWorld windy_world;

    // Test
    // 1. Normal behaviour when no wind.
    grid::Position no_wind_pos{5, 1};
    const rl::State& no_wind_state = windy_world.pos_to_state(no_wind_pos);
    for(grid::Direction d : grid::directions) {
        grid::Position adj = no_wind_pos.adj(d);
        const rl::State& expected = windy_world.pos_to_state(adj);
        const rl::State& actual =
                windy_world.next_state(no_wind_state, windy_world.dir_to_action(d)).next_state;
        ASSERT_EQ(expected, actual);
    }

    // 2. Wind applies before move.
    // pos_a has 1 wind, and pos_b, to the right, has 2 wind.
    // Moving right from pos_a should move 1 right and 1 up (not 1 right and 2 up).
    grid::Position pos_a{5, 3};
    // Moving Up means -ve y change:
    grid::Position pos_b{4, 4};
    const rl::State& state_a = windy_world.pos_to_state(pos_a);
    const rl::State& state_b = windy_world.pos_to_state(pos_b);
    ASSERT_EQ(state_b, windy_world.next_state(
                          state_a, windy_world.dir_to_action(grid::Direction::RIGHT)).next_state);

    // 3. Wind pushes into boundary.
    // (0,5) -> right -> (0, 5), when in upward wind.
    grid::Position pos_c{0, 5};
    grid::Position pos_d{0, 6};
    const rl::State& state_c = windy_world.pos_to_state(pos_c);
    const rl::State& state_d = windy_world.pos_to_state(pos_d);
    ASSERT_EQ(state_d, windy_world.next_state(
                            state_c, windy_world.dir_to_action(grid::Direction::RIGHT)).next_state);
}
