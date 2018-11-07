#pragma  once

#include <stdexcept>
#include <sstream>
#include <unordered_map>

#include "TestEnvironment.h"
#include "rl/MappedEnvironment.h"
#include "rl/impl/Environment.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_cdf.h"
#include "rl/GridWorld.h"
#include "grid/Grid.h"
#include "util/random.h"

namespace rl {
namespace test {

class Exercise4_1 : public TestEnvironment {
public:
    static const int GRID_WIDTH = 4;
    static const int GRID_HEIGHT = 4;
    using GridType = GridWorld<GRID_HEIGHT, GRID_WIDTH>;

    static constexpr double expected_values[] =
            {0.0, -14, -20, -22,
             -14, -18, -20, -20,
             -20, -20, -18, -14,
             -22, -20, -14, 0.0};

    std::string name() const override {
        return "Sutton & Barto exercise 4.1";
    }

    double required_discount_rate() const override {
        return 1.0;
    }

    double required_delta_threshold() const override {
        return 1e-2;
    }

    const MappedEnvironment& env() const override {
        return grid_world_.environment();
    }

    OptimalActions optimal_actions(const State& from_state) const override {
        OptimalActions ans;
        std::transform(
            std::begin(optimal_actions_[from_state.id()]),
            std::end(optimal_actions_[from_state.id()]),
            std::inserter(ans, std::end(ans)),
            [this](int dir) {
                return grid_world_.dir_to_action(grid::directions[dir]).id();
            }
            );
        return ans;
    }

    // TODO: This method can be removed at some point.
    GridType& grid_world() {
        return grid_world_;
    }

private:
    static GridType create_grid_world() {
        // Setup.
        rl::GridWorld<GRID_HEIGHT, GRID_WIDTH> grid_world;
        const grid::Position top_left{0, 0};
        const grid::Position bottom_right{GRID_HEIGHT-1, GRID_WIDTH-1};
        grid_world.environment().mark_as_end_state(grid_world.pos_to_state(top_left));
        grid_world.environment().mark_as_end_state(grid_world.pos_to_state(bottom_right));
        grid_world.environment().set_all_rewards_to(-1.0);
        grid_world.environment().build_distribution_tree();
        return grid_world;
    }

private:
    GridType grid_world_ = create_grid_world();

    static const std::unordered_set<int> optimal_actions_[GRID_WIDTH * GRID_HEIGHT];
};


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
        enum class Location {LOC1, LOC2};
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
        double poisson_pdf_cache [MEAN_RANGE][MAX_CAR_COUNT + 1];
        double poisson_cdf_cache [MEAN_RANGE][MAX_CAR_COUNT + 1];

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
                if(i < 0) {
                    action_name << " from location 2";
                } else if(i > 0) {
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
        const State &state(int cars_in_loc1, int cars_in_loc2) const {
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
            if(loc == Location::LOC1) {
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
            return poisson_pdf_cache[mean-MIN_MEAN][x];
        }

        double upper_poisson_cdf(int greater_equal_than, int mean) const {
            Ensures(greater_equal_than >= 0);
            if(greater_equal_than == 0) {
                return 1.0;
            } else {
                // return gsl_cdf_poisson_Q(greater_equal_than-1, mean);
                return poisson_cdf_cache[mean-MIN_MEAN][greater_equal_than-1];
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
                if(new_car_count == MAX_CAR_COUNT) {
                    // Upper CDF.
                    returned_prob = upper_poisson_cdf(returned, return_mean);
                } else {
                    Ensures(returned >= 0);
                    returned_prob = poisson_pdf(returned, return_mean);
                }
                double rented_prob = 0;
                if(rented == prev_car_count) {
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
        transition_list(const State &from_state, const Action &action) const override {
            Expects(is_action_allowed(from_state, action));
            ResponseDistribution ans{};
            int loc1_start = cars_in_loc_1(from_state) + change_in_car_count(action, Location::LOC1);
            int loc2_start = cars_in_loc_2(from_state) + change_in_car_count(action, Location::LOC2);
            // Transfer cost. Let's just keep this here.
            int transfer_cost = std::abs(change_in_car_count(action, Location::LOC1)) * TRANSFER_COST;
            // Calculate all possible transitions from this state.
            for (ID loc1_end = 0; loc1_end <= MAX_CAR_COUNT; loc1_end++) {
                TransitionPart t1 = possibilities(loc1_start, loc1_end,
                        LOC1_RENTAL_MEAN, LOC1_RETURN_MEAN);
                for (ID loc2_end = 0; loc2_end <= MAX_CAR_COUNT; loc2_end++) {
                    TransitionPart t2 = possibilities(loc2_start, loc2_end,
                                                              LOC2_RENTAL_MEAN, LOC2_RETURN_MEAN);
                    const State &next_state = state(loc1_end, loc2_end);
                    double probability = t1.probability * t2.probability;
                    // We are going to ignore transitions with very small transition possibilities.
                    if(probability < MIN_PROB) {
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
            for(int mu = 0; mu < MEAN_RANGE; mu++) {
                for(int j = 0; j <= MAX_CAR_COUNT; j++) {
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
          // FIXME: (17, 8) should be: 0. Making the tests assert incorrect behaviour until fixed.
          {5,  5,  5,  4,  3,  2,  2,  1,  1/*0*/,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 17
          {5,  5,  5,  4,  3,  3,  2,  2,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 18
          {5,  5,  5,  4,  4,  3,  3,  2,  2,  2,  2,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0}, // 19
          {5,  5,  5,  5,  4,  4,  3,  3,  3,  3,  2,  2,  2,  2,  2,  1,  1,  1,  0,  0,  0}  // 20
        };
};

class Exercise5_1 : public TestEnvironment {
public:

    class BlackjackEnvironment : public rl::impl::Environment {

    public:
        // The hit/stick choice is always 'hit' for sums lower than 12 as no drawn card would cause
        // the player to go bust. Therefore, we consider the decision problem to start from the
        // point here the card sum is greater than or equal to 12.
        static const int MIN_SUM = 12;
        // If the sum is greater than 21, the player has 'gone bust', which represents the losing
        // end state. 21 is not an end state, as the player could win or draw depending on the
        // dealer's hand.
        static constexpr int MAX_SUM = 21;
        static constexpr int STATE_COUNT = 200;
        static constexpr int ACE = 1;
        static constexpr int TEN = 10;
        static constexpr double WIN_REWARD =   1;
        static constexpr double DRAW_REWARD =  0;
        static constexpr double LOSS_REWARD = -1;
        static constexpr int DEALER_STICK_THRESHOLD = 17;

        enum class BlackjackAction {HIT=0, STICK=1};
        struct BlackjackState {
            int player_sum;
            bool usable_ace;
            int dealer_card;

            bool operator==(const BlackjackState& other) const {
                return player_sum == other.player_sum and
                       usable_ace == other.usable_ace and
                       dealer_card == other.dealer_card;
            }
        };

        BlackjackEnvironment() {
            // Create the 200 (non-end) states.
            // The dealer's one showing card can be one of the ten cards: [ace, 10].
            for(int dealer_card = ACE; dealer_card <= TEN; dealer_card++) {
                for(int player_sum = MIN_SUM; player_sum <= MAX_SUM; player_sum++) {
                    for(bool usable_ace : {false, true}) {
                        std::stringstream state_name;
                        state_name << "P (sum: " << player_sum << ", using ace:" << usable_ace
                                   << "), ";
                        state_name << "D (card: ";
                        if(dealer_card == ACE) {
                            state_name << "ace)";
                        } else {
                            state_name << dealer_card << ")";
                        }
                        ID id = state_id({player_sum, usable_ace, dealer_card});
                        add_state(std::make_unique<State>(id, state_name.str()));
                        Ensures(states_.back().get()->id() == id);
                        id_to_blackjack_state.insert(
                                {id, BlackjackState{player_sum, usable_ace, dealer_card}});
                    }
                }
            }
            // Create the 3 end states.
            win_state_  = &add_end_state(std::make_unique<State>(state_count(), "win"));
            draw_state_ = &add_end_state(std::make_unique<State>(state_count(), "draw"));
            loss_state_ = &add_end_state(std::make_unique<State>(state_count(), "lose"));
            CHECK_EQ(203, state_count());

            // Create the 2 actions.
            add_action(std::make_unique<Action>(action_id(BlackjackAction::HIT), "hit"));
            add_action(std::make_unique<Action>(action_id(BlackjackAction::STICK), "stick"));
            CHECK_EQ(2, action_count());
            validate();
        }

        const State& win_state() const {
            return *CHECK_NOTNULL(win_state_);
        }

        const State& draw_state() const {
            return *CHECK_NOTNULL(draw_state_);
        }

        const State& loss_state() const {
            return *CHECK_NOTNULL(loss_state_);
        }

        ID action_id(BlackjackAction hit_or_stick) const {
            return static_cast<ID>(hit_or_stick);
        }

        using Environment::state; // To prevent hiding.
        const State& state(const BlackjackState& blackjack_state) const {
            return state(state_id(blackjack_state));
        }

        using Environment::action; // To prevent hiding.
        const Action& action(const BlackjackAction& blackjack_action) const {
            return action(action_id(blackjack_action));
        }

        BlackjackAction blackjack_action(const Action& action) const {
            return static_cast<BlackjackAction>(action.id());
        }

        ID state_id(BlackjackState state) const {
            // We have a R^R^R space which we wish to map (1-1) to the natural numbers.
            const int dim_1_max = 2; // usable_ace = {false, true}.
            const int dim_2_max = (MAX_SUM - MIN_SUM) + 1; // player sum
            ID id = (state.dealer_card - 1) * (dim_1_max * dim_2_max) +
                    (state.player_sum - MIN_SUM) * dim_1_max + static_cast<int>(state.usable_ace);
            Ensures(id < STATE_COUNT);
            Ensures(id >= 0);
            return id;
        }

        BlackjackState blackjack_state(const State& state) const {
            // Just being lazy and using a map for this direction.
            Expects(state != win_state());
            Expects(state != draw_state());
            Expects(state != loss_state());
            return id_to_blackjack_state.at(state.id());
        }

        static int card_value(int card_id) {
            if(card_id == ACE) {
                return 11;
            } else {
                return card_id;
            }
        }

        static int random_card() {
            int random_card = util::random_in_range(ACE, TEN + 3 + 1);
            random_card = std::min(TEN, random_card);
            return random_card;
        }

        static double card_chance(int card) {
            if(card == TEN) {
                return 4.0/13.0;
            } else {
                return 1.0/13.0;
            }
        }

        static int simulate_dealer_turn(int visible_card) {
            int dealer_hidden_card = random_card();
            int sum = card_value(dealer_hidden_card) + card_value(visible_card);
            bool has_ace = (visible_card == ACE) or (dealer_hidden_card == ACE);
            CHECK_LE(sum, card_value(ACE)*2) << "The maximum sum comes from 2 aces";
            // If there are two aces, only 1 will be used.
            if(sum == card_value(ACE)*2) {
                sum = card_value(ACE) + 1;
            }
            while(sum < DEALER_STICK_THRESHOLD) {
                int next_card = random_card();
                // Use a dummy BlackjackState object, treating the dealer as the player.
                BlackjackState next = calculate_next_state({sum, has_ace, 0}, next_card);
                sum = next.player_sum;
                has_ace = next.usable_ace;
            }
            return sum;
        }

        Response next_state(const State& from_state, const Action& action) const override {
            const BlackjackState state_data = blackjack_state(from_state);
            switch(blackjack_action(action)) {
                case BlackjackAction::STICK:
                    return stick_response(state_data);
                case BlackjackAction::HIT:
                    return hit_response(state_data);
            }
            // This should never be reached.
            CHECK(false);
            // TODO: silence compiler-warning for possible void return.
        }

        // TODO: make some methods private.
        static BlackjackState calculate_next_state(const BlackjackState& current, int card_hit) {
            BlackjackState next(current);
            int ace_count = static_cast<int>(current.usable_ace);
            if(card_hit == ACE) {
                ace_count++;
            }
            next.player_sum += card_value(card_hit);
            while(next.player_sum > MAX_SUM and ace_count) {
                next.player_sum = revert_ace(next.player_sum);
                ace_count--;
                CHECK_LE(next.player_sum, MAX_SUM + 1)
                    << "The maximum sum after changing an ace from 11->1 is 22 "
                       "(when the sum is 21 and an ace is received).";
            }
            CHECK(ace_count <= 1 and ace_count >= 0)
                << "The player can't have 2 aces at 11 points each.";
            next.usable_ace = (ace_count == 1);
            return next;
        }

        Response hit_response(const BlackjackState& state_data) const {
            int next_card = random_card();
            BlackjackState next = calculate_next_state(state_data, next_card);
            if(next.player_sum > MAX_SUM) {
                // TODO: sort out transient reward ID.
                // TODO: make the Response's weight not a requirement.
                return Response{loss_state(), Reward{-1, LOSS_REWARD}, 1.0};
            } else {
                return Response{state(state_id(next)), Reward{-1, 0}, 1.0};
            }
        }

        Response stick_response(const BlackjackState& state_data) const {
            // Determine if it is a win, loose or draw.
            // 1. Calculate dealer's sum.
            int dealer_sum = simulate_dealer_turn(state_data.dealer_card);

            // 2. Player's sum.
            int player_sum = state_data.player_sum;
            // It would be impossible to 'stick' from a state where the player has > 21 points.
            Ensures(player_sum <= MAX_SUM);

            double reward_value = 0;
            const State* end_state = nullptr;
            if(dealer_sum > MAX_SUM or dealer_sum < player_sum) {
                end_state = &win_state();
                reward_value = WIN_REWARD;
            } else if(dealer_sum > player_sum) {
                end_state = &loss_state();
                reward_value = LOSS_REWARD;
            } else {
                Ensures(dealer_sum == player_sum);
                end_state = &draw_state();
                reward_value = DRAW_REWARD;
            }
            // TODO: sort out the id for transient rewards.
            ID invalid_id = -1;
            Response r{*CHECK_NOTNULL(end_state),
                       Reward(invalid_id, reward_value), 1.0};
            return r;
        }

        bool is_action_allowed(const State& from_state, const Action& a) const override {
            // Both hit and stick are allowed at any time.
            return true;
        }

        static int revert_ace(int previous_sum) {
            CHECK_GE(previous_sum, ACE);
            return previous_sum - card_value(ACE) + 1;
        }

        ResponseDistribution
        transition_list(const State& from_state, const Action& action) const override {
            ResponseDistribution ans;
            const BlackjackState state_data = blackjack_state(from_state);
            EndingWeights counts{};
            switch(blackjack_action(action)) {
                case BlackjackAction::HIT:
                    // Choose a card. 1/13 chance of getting each card [Ace, 9] and 4/13 for 10.
                    for(int card = ACE; card <= TEN; card++) {
                        BlackjackState next = calculate_next_state(state_data, card);
                        if(next.player_sum > MAX_SUM) {
                            // The player loses.
                            // There are multiple cards that could result in a loss, so count them
                            // before adding a response.
                            counts.loss += card_chance(card);
                            continue;
                        } else {
                            const double reward = 0;
                            ans.add_response(Response{state(state_id(next)),
                                                      Reward(-1, reward),
                                                      card_chance(card)});
                        }
                    }
                    break;
                case BlackjackAction::STICK:
                    bool dealer_has_ace = state_data.dealer_card == ACE;
                    tally_endings(state_data.player_sum, card_value(state_data.dealer_card),
                            dealer_has_ace, counts, 1.0);
                    break;
            }
            // Add the end states (only if there is a transition possibility).
            if(counts.win) {
                ans.add_response(Response{win_state(),
                                          Reward(-1, WIN_REWARD),
                                          static_cast<Weight>(counts.win)});
            }
            if(counts.draw) {
                ans.add_response(Response{draw_state(),
                                          Reward(-1, DRAW_REWARD),
                                          static_cast<Weight>(counts.draw)});
            }
            if(counts.loss) {
                ans.add_response(Response{loss_state(),
                                          Reward(-1, LOSS_REWARD),
                                          static_cast<Weight>(counts.loss)});
            }
            return ans;
        }
    private:
        struct EndingWeights {
            double win = 0.0;
            double draw = 0.0;
            double loss = 0.0;
        };
        // This function is involves traversing a tree that has natural repetition. The function
        // can be tweaked to use DP memorization if the function becomes a performance bottleneck.
        void tally_endings(const int player_sum, const int dealer_sum, const bool dealer_usable_ace,
                           EndingWeights& counts, const double parent_prob) const {
            // Handle the dealer going bust before calling this method.
            Expects(dealer_sum <= MAX_SUM);
            // Shortcut exit if the dealer already has a higher sum.
            if(dealer_sum > player_sum) {
                counts.loss += parent_prob;
                return;
            }
            if(dealer_sum >= DEALER_STICK_THRESHOLD) {
                // Dealer will stick.
                if(dealer_sum < player_sum) {
                    counts.win += parent_prob;
                } else if(dealer_sum > player_sum) {
                    counts.loss += parent_prob;
                } else {
                    CHECK_EQ(dealer_sum, player_sum);
                    counts.draw += parent_prob;
                }
            } else {
                // Dealer will hit.
                for(int card = ACE; card <= TEN; card++) {
                    double prob = parent_prob * card_chance(card);
                    // Use a dummy BlackjackState object, treating the dealer as the player.
                    BlackjackState after_hit = calculate_next_state({dealer_sum, dealer_usable_ace,
                                                                     0}, card);
                    int sum = after_hit.player_sum;
                    bool usable_ace = after_hit.usable_ace;
                    if(sum > MAX_SUM) {
                        counts.win += prob;
                    } else {
                        tally_endings(player_sum, sum, usable_ace, counts, prob);
                    }
                }
            }
        }

    private:
        std::unordered_map<ID, BlackjackState> id_to_blackjack_state;
        const State* win_state_ = nullptr;
        const State* draw_state_ = nullptr;
        const State* loss_state_ = nullptr;
    };

    std::string name() const override {
        return "Sutton & Barto exercise 5.1";
    }

    const Environment& env() const override {
        return env_;
    }

    double required_discount_rate() const override {
        return 1.0;
    }

    double required_delta_threshold() const override {
        return 1e-5;
    }

    OptimalActions optimal_actions(const State& from_state) const override {
        if(env_.is_end_state(from_state)) {
            return OptimalActions{};
        }
        BlackjackEnvironment::BlackjackAction action =
                optimal_action(env_.blackjack_state(from_state));
        return OptimalActions{env_.action_id(action)};
    }

    static BlackjackEnvironment::BlackjackAction
    optimal_action(BlackjackEnvironment::BlackjackState from_state) {
        // Transcribed from (Sutton & Barto, 2018) p100.
        Expects(from_state.player_sum >= 12);
        Expects(from_state.player_sum <= 21);
        Expects(from_state.dealer_card >= BlackjackEnvironment::ACE);
        Expects(from_state.dealer_card <= BlackjackEnvironment::TEN);
        BlackjackEnvironment::BlackjackAction h = BlackjackEnvironment::BlackjackAction::HIT;
        BlackjackEnvironment::BlackjackAction s = BlackjackEnvironment::BlackjackAction::STICK;
        BlackjackEnvironment::BlackjackAction ans;
        if(from_state.usable_ace) {
            if(from_state.player_sum <= 17) {
                ans = h;
            } else if(from_state.player_sum >= 19) {
                ans = s;
            } else {
                CHECK_EQ(from_state.player_sum, 18);
                if(BlackjackEnvironment::card_value(from_state.dealer_card) >= 9) {
                    ans = h;
                } else {
                    ans = s;
                }
            }
        } else {
            if(from_state.player_sum >= 17) {
                ans = s;
            } else if(BlackjackEnvironment::card_value(from_state.dealer_card) >= 7) {
                ans = h;
            } else if(from_state.player_sum >= 13) {
                ans = s;
            } else if(from_state.player_sum == 12 and
                         ((from_state.dealer_card == 4) or
                          (from_state.dealer_card == 5) or
                          (from_state.dealer_card == 6))
                      ) {
                ans = s;
            } else {
                ans = h;
            }
        }
        return ans;
    }

private:
    BlackjackEnvironment env_;
};
} // namespace test
} // namespace rl

