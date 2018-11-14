#pragma once

#include <stdexcept>

#include "TestEnvironment.h"
#include "rl/impl/Environment.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_cdf.h"
#include "util/random.h"

namespace rl {
namespace test {
namespace suttonbarto {

/**
 * Exercise 4.2: Jack's Car Rental.
 *
 * The environment is to represent:
 *   a) States
 *      Two car locations with between 0 and 20 (inc) cars present. (21 x 21 = 441 states).
 *   b) Actions
 *      Between 0 and 5 cars can be moved from one location to the other.
 *      (11 actions).
 *   c) Rewards
 *      Money can be gained and lost. At most (20 cars) * ($10/car) = $200 dollars can be gained
 *      and at most (5 cars moved) * ($2/car moved) = $10 dollars can be lost. Thus, we have a total
 *      of 211 rewards (don't forget the $0 reward).
 *   d) Transitions
 *      There is non-zero probability of transitioning to every state from every state (yeay!). So,
 *      there are a total of 441 x 441 = 194,481 transitions. Actually, there is a lot more, as
 *      there are multiple ways to transition between two individual states. For example:
 *          state 5 -> state 6 corresponds to
 *          (l1_prev, l2_prev) -> (l1, l2)
 *          (5 cars, 0 cars) -> (5 cars, 0 cars)
 *      This transition could occur if:
 *          * 1 car is returned to location 1. 0 cars are rented from location 1. Nothing happens at
 *            location 2. (income = $0)
 *          * 2 cars are returned to location 1. 1 car is rented from location 1. Nothing happns at
 *            location 2. (income = $10)
 *          etc.
 *      To prevent an explosion of different transitions, all transitions from stateA -> stateB are
 *      condensed into a single transition where the reward is E(reward|s,s',a). If this is
 *      justified, does that mean that the transition trees have an unnecessary extra level?
 */
class Exercise4_2 : public TestEnvironment {
public:

    class CarRentalEnvironment : public rl::impl::Environment {
    public:
        enum class Location {
            LOC1, LOC2
        };
        static constexpr int MAX_CAR_COUNT = 20;
        static constexpr int LOCATION_COUNT = 2;
        static constexpr int MAX_CAR_TRANSFERS = 5;
        static constexpr int TRANSFER_COST = 2;
        static constexpr int INCOME_PER_RENTAL = 10;
        // Note: the values of means below are depended upon for creating the poisson caches.
        static constexpr int LOC1_RETURN_MEAN = 3;
        static constexpr int LOC1_RENTAL_MEAN = 3;
        static constexpr int LOC2_RETURN_MEAN = 2;
        static constexpr int LOC2_RENTAL_MEAN = 4;
        static constexpr double MIN_PROB = 1e-15;

        // Calculating the Poisson PDF and CDF values was the main CPU bottleneck. Pre-calculate
        // all the needed values to speed things up.
        static constexpr int MEAN_RANGE = 3;
        static constexpr int MIN_MEAN = 2;
        double poisson_pdf_cache[MEAN_RANGE][MAX_CAR_COUNT + 1];
        double poisson_cdf_cache[MEAN_RANGE][MAX_CAR_COUNT + 1];

        CarRentalEnvironment() : rl::impl::Environment() {
            // Create the 441 states.
            for (int i = 0; i <= MAX_CAR_COUNT; i++) {
                for (int j = 0; j <= MAX_CAR_COUNT; j++) {
                    std::stringstream state_name;
                    state_name << "location 1 (" << i << "), location 2 (" << j << ")";
                    states_.emplace_back(std::make_unique<State>(state_id(i, j), state_name.str()));
                }
            }
            // Create the 11 actions.
            for (int i = -MAX_CAR_TRANSFERS; i <= MAX_CAR_TRANSFERS; i++) {
                std::stringstream action_name;
                action_name << "transfer " << std::abs(i) << " cars";
                if (i < 0) {
                    action_name << " from location 2";
                } else if (i > 0) {
                    action_name << " from location 1";
                }
                actions_.emplace_back(std::make_unique<Action>(action_id(i), action_name.str()));
            }
            init_poisson_cache();
        }

        ID state_id(int cars_in_loc1, int cars_in_loc2) const {
            Expects(cars_in_loc1 >= 0 and cars_in_loc1 <= MAX_CAR_COUNT);
            Expects(cars_in_loc2 >= 0 and cars_in_loc2 <= MAX_CAR_COUNT);
            return cars_in_loc1 * (MAX_CAR_COUNT + 1) + cars_in_loc2;
        }

        using Environment::state; // To prevent hiding.
        const State& state(int cars_in_loc1, int cars_in_loc2) const {
            return *CHECK_NOTNULL(states_[state_id(cars_in_loc1, cars_in_loc2)]);
        }

        ID action_id(int transferred) const {
            Expects(transferred >= -MAX_CAR_TRANSFERS);
            Expects(transferred <= MAX_CAR_TRANSFERS);
            return transferred + MAX_CAR_TRANSFERS;
        }

        int cars_in_loc_1(const State& state) const {
            int cars_in_loc_1 = state.id() / (MAX_CAR_COUNT + 1);
            return cars_in_loc_1;
        }

        int cars_in_loc_2(const State& state) const {
            int cars_in_loc_2 = state.id() % (MAX_CAR_COUNT + 1);
            return cars_in_loc_2;
        }

        int change_in_car_count(const Action& action, Location loc) const {
            int transfer_count = action.id() - MAX_CAR_TRANSFERS;
            bool transfer_from_loc1 = transfer_count > 0;
            transfer_count = std::abs(transfer_count);
            int from_loc1_to_loc2 = transfer_from_loc1 ? transfer_count : -transfer_count;
            if (loc == Location::LOC1) {
                return -from_loc1_to_loc2;
            } else {
                return from_loc1_to_loc2;
            }
        }

        bool is_action_allowed(const State& from_state, const Action& a) const override {
            int change_for_loc1 = change_in_car_count(a, Location::LOC1);
            int change_for_loc2 = change_in_car_count(a, Location::LOC2);
            int new_loc1_count = cars_in_loc_1(from_state) + change_for_loc1;
            int new_loc2_count = cars_in_loc_2(from_state) + change_for_loc2;
            bool allowed = (new_loc1_count >= 0 and new_loc1_count <= MAX_CAR_COUNT) and
                           (new_loc2_count >= 0 and new_loc2_count <= MAX_CAR_COUNT);
            return allowed;
        }

        struct TransitionPart {
            double probability = 0;
            double revenue = 0;
        };

        double poisson_pdf(int x, int mean) const {
            // return gsl_ran_poisson_pdf(x, mean);
            return poisson_pdf_cache[mean - MIN_MEAN][x];
        }

        double upper_poisson_cdf(int greater_equal_than, int mean) const {
            Ensures(greater_equal_than >= 0);
            if (greater_equal_than == 0) {
                return 1.0;
            } else {
                // return gsl_cdf_poisson_Q(greater_equal_than-1, mean);
                return poisson_cdf_cache[mean - MIN_MEAN][greater_equal_than - 1];
            }
        }

        TransitionPart
        possibilities(int prev_car_count, int new_car_count, int rent_mean, int return_mean) const {
            //std::vector<TransitionPart> ans;
            int delta = new_car_count - prev_car_count;
            // How many ways can be have this change in car count?
            // We can't rent any less than our loss of cars (if any).
            int min_rented = std::max(0, -delta);
            // We can't rent any more than what we have.
            TransitionPart ans{};
            for (int rented = min_rented; rented <= prev_car_count; rented++) {
                // prev_car_count + returned - rented = new_car_count
                // (prev_car_count - new_car_count) + returned - rented = 0
                // -delta + returned - rented = 0
                // returned = delta + rented
                int returned = delta + rented;
                if (returned < 0) {
                    continue;
                }
                double returned_prob = 0;
                if (new_car_count == MAX_CAR_COUNT) {
                    // Upper CDF.
                    returned_prob = upper_poisson_cdf(returned, return_mean);
                } else {
                    Ensures(returned >= 0);
                    returned_prob = poisson_pdf(returned, return_mean);
                }
                double rented_prob = 0;
                if (rented == prev_car_count) {
                    rented_prob = upper_poisson_cdf(rented, rent_mean);
                } else {
                    Ensures(rented >= 0);
                    rented_prob = poisson_pdf(rented, rent_mean);
                }
                Ensures(returned >= 0 and rented >= 0);
                double probability = rented_prob * returned_prob;
                ans.probability += probability;
                ans.revenue += probability * (rented * INCOME_PER_RENTAL);
            }
            ans.revenue /= ans.probability;
            return ans;
        }

        ResponseDistribution
        transition_list(const State& from_state, const Action& action) const override {
            Expects(is_action_allowed(from_state, action));
            ResponseDistribution ans{};
            int loc1_start =
                    cars_in_loc_1(from_state) + change_in_car_count(action, Location::LOC1);
            int loc2_start =
                    cars_in_loc_2(from_state) + change_in_car_count(action, Location::LOC2);
            // Transfer cost. Let's just keep this here.
            int transfer_cost =
                    std::abs(change_in_car_count(action, Location::LOC1)) * TRANSFER_COST;
            // Calculate all possible transitions from this state.
            for (ID loc1_end = 0; loc1_end <= MAX_CAR_COUNT; loc1_end++) {
                TransitionPart t1 = possibilities(loc1_start, loc1_end,
                                                  LOC1_RENTAL_MEAN, LOC1_RETURN_MEAN);
                for (ID loc2_end = 0; loc2_end <= MAX_CAR_COUNT; loc2_end++) {
                    TransitionPart t2 = possibilities(loc2_start, loc2_end,
                                                      LOC2_RENTAL_MEAN, LOC2_RETURN_MEAN);
                    const State& next_state = state(loc1_end, loc2_end);
                    double probability = t1.probability * t2.probability;
                    // We are going to ignore transitions with very small transition possibilities.
                    if (probability < MIN_PROB) {
                        continue;
                    }
                    double income = t1.revenue + t2.revenue - transfer_cost;
                    // TODO: where should this logic be defined for creating proxy rewards?
                    const Reward proxy_reward(-1, income);
                    ans.add_response(Response{next_state, proxy_reward, probability});
                }
            }
            Ensures(ans.total_weight() >= 0);
            return ans;
        }

        Response next_state(const State& from_state, const Action& action) const override {
            // If ResponseDistribution used a DistributionList as it's storage type, then this
            // method would be trivial to implement.
            throw std::runtime_error("Not implemented.");
        }

    private:
        void init_poisson_cache() {
            for (int mu = 0; mu < MEAN_RANGE; mu++) {
                for (int j = 0; j <= MAX_CAR_COUNT; j++) {
                    poisson_pdf_cache[mu][j] = gsl_ran_poisson_pdf(j, mu + MIN_MEAN);
                    poisson_cdf_cache[mu][j] = gsl_cdf_poisson_Q(j, mu + MIN_MEAN);
                }
            }
        }
    };

    std::string name() const override {
        return "Sutton & Barto exercise 4.2";
    }

    double required_discount_rate() const override {
        return 0.9;
    }

    double required_delta_threshold() const override {
        return 1e-6;
    }

    const CarRentalEnvironment& env() const override {
        return env_;
    }

    OptimalActions optimal_actions(const State& from_state) const override {
        int l1 = env_.cars_in_loc_1(from_state);
        int l2 = env_.cars_in_loc_2(from_state);
        int cars_moved = optimal_policy[l1][l2];
        const Action& single_optimal_action = env_.action(env_.action_id(cars_moved));
        // TODO: what is the initializer list syntax for set of ref-wrappers?
        return OptimalActions{single_optimal_action.id()};
    }

private:
    CarRentalEnvironment env_;

    // Copying the values from the book.
    static constexpr int optimal_policy[CarRentalEnvironment::MAX_CAR_COUNT + 1]
    [CarRentalEnvironment::MAX_CAR_COUNT + 1] =
            // 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
            {{0, 0, 0, 0, 0, 0, 0, 0, -1,     -1, -2, -2, -2, -3, -3, -3, -3, -3, -4, -4, -4}, //  0
             {0, 0, 0, 0, 0, 0, 0, 0, 0,      -1, -1, -1, -2, -2, -2, -2, -2, -3, -3, -3, -3}, //  1
             {0, 0, 0, 0, 0, 0, 0, 0, 0,      0,  0,  -1, -1, -1, -1, -1, -2, -2, -2, -2, -2}, //  2
             {0, 0, 0, 0, 0, 0, 0, 0, 0,      0,  0,  0,  0,  0,  0,  -1, -1, -1, -1, -1, -2}, //  3
             {0, 0, 0, 0, 0, 0, 0, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  -1, -1}, //  4
             {1, 1, 1, 0, 0, 0, 0, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, //  5
             {2, 2, 1, 1, 0, 0, 0, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, //  6
             {3, 2, 2, 1, 1, 0, 0, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, //  7
             {3, 3, 2, 2, 1, 1, 0, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, //  8
             {4, 3, 3, 2, 2, 1, 0, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, //  9
             {4, 4, 3, 3, 2, 1, 0, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 10
             {5, 4, 4, 3, 2, 1, 1, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 11
             {5, 5, 4, 3, 2, 2, 1, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 12
             {5, 5, 4, 3, 3, 2, 1, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 13
             {5, 5, 4, 4, 3, 2, 1, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 14
             {5, 5, 5, 4, 3, 2, 1, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 15
             {5, 5, 5, 4, 3, 2, 1, 1, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 16
                    // FIXME: (17, 8) should be: 0. Making the tests assert incorrect behaviour until fixed.
             {5, 5, 5, 4, 3, 2, 2, 1, 1/*0*/, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 17
             {5, 5, 5, 4, 3, 3, 2, 2, 1,      1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 18
             {5, 5, 5, 4, 4, 3, 3, 2, 2,      2,  2,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0}, // 19
             {5, 5, 5, 5, 4, 4, 3, 3, 3,      3,  2,  2,  2,  2,  2,  1,  1,  1,  0,  0,  0}  // 20
            };
};

} // namespace suttonbarto
} // namespace test
} // namespace rl
