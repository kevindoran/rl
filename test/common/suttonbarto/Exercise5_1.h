#pragma  once

#include <unordered_map>

#include "TestEnvironment.h"
#include "rl/impl/Environment.h"
#include "util/random.h"

namespace rl {
namespace test {
namespace suttonbarto {



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
        static constexpr double WIN_REWARD = 1;
        static constexpr double DRAW_REWARD = 0;
        static constexpr double LOSS_REWARD = -1;
        static constexpr int DEALER_STICK_THRESHOLD = 17;

        enum class BlackjackAction {
            HIT = 0, STICK = 1
        };

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
            for (int dealer_card = ACE; dealer_card <= TEN; dealer_card++) {
                for (int player_sum = MIN_SUM; player_sum <= MAX_SUM; player_sum++) {
                    for (bool usable_ace : {false, true}) {
                        std::stringstream state_name;
                        state_name << "P (sum: " << player_sum << ", using ace:" << usable_ace
                                   << "), ";
                        state_name << "D (card: ";
                        if (dealer_card == ACE) {
                            state_name << "ace)";
                        } else {
                            state_name << dealer_card << ")";
                        }
                        ID id = state_id({player_sum, usable_ace, dealer_card});
                        add_state(state_name.str());
                        Ensures(states_.back().get()->id() == id);
                        id_to_blackjack_state.insert(
                                {id, BlackjackState{player_sum, usable_ace, dealer_card}});
                    }
                }
            }
            // Create the 3 end states.
            win_state_ = &add_end_state("win");
            draw_state_ = &add_end_state("draw");
            loss_state_ = &add_end_state("lose");
            CHECK_EQ(203, state_count());

            // Create the 2 actions.
            add_action("hit");
            CHECK_EQ(actions_.back()->id(), action_id(BlackjackAction::HIT));
            add_action("stick");
            CHECK_EQ(actions_.back()->id(), action_id(BlackjackAction::STICK));
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
            if (card_id == ACE) {
                return 11;
            } else {
                return card_id;
            }
        }

        static int random_card() {
            int random_card = util::random::random_in_range(ACE, TEN + 3 + 1);
            random_card = std::min(TEN, random_card);
            return random_card;
        }

        static double card_chance(int card) {
            if (card == TEN) {
                return 4.0 / 13.0;
            } else {
                return 1.0 / 13.0;
            }
        }

        static int simulate_dealer_turn(int visible_card) {
            int dealer_hidden_card = random_card();
            int sum = card_value(dealer_hidden_card) + card_value(visible_card);
            bool has_ace = (visible_card == ACE) or (dealer_hidden_card == ACE);
            CHECK_LE(sum, card_value(ACE) * 2) << "The maximum sum comes from 2 aces";
            // If there are two aces, only 1 will be used.
            if (sum == card_value(ACE) * 2) {
                sum = card_value(ACE) + 1;
            }
            while (sum < DEALER_STICK_THRESHOLD) {
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
            switch (blackjack_action(action)) {
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
            if (card_hit == ACE) {
                ace_count++;
            }
            next.player_sum += card_value(card_hit);
            while (next.player_sum > MAX_SUM and ace_count) {
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
            if (next.player_sum > MAX_SUM) {
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
            if (dealer_sum > MAX_SUM or dealer_sum < player_sum) {
                end_state = &win_state();
                reward_value = WIN_REWARD;
            } else if (dealer_sum > player_sum) {
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
            switch (blackjack_action(action)) {
                case BlackjackAction::HIT:
                    // Choose a card. 1/13 chance of getting each card [Ace, 9] and 4/13 for 10.
                    for (int card = ACE; card <= TEN; card++) {
                        BlackjackState next = calculate_next_state(state_data, card);
                        if (next.player_sum > MAX_SUM) {
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
            if (counts.win) {
                ans.add_response(Response{win_state(),
                                          Reward(-1, WIN_REWARD),
                                          static_cast<Weight>(counts.win)});
            }
            if (counts.draw) {
                ans.add_response(Response{draw_state(),
                                          Reward(-1, DRAW_REWARD),
                                          static_cast<Weight>(counts.draw)});
            }
            if (counts.loss) {
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
            if (dealer_sum > player_sum) {
                counts.loss += parent_prob;
                return;
            }
            if (dealer_sum >= DEALER_STICK_THRESHOLD) {
                // Dealer will stick.
                if (dealer_sum < player_sum) {
                    counts.win += parent_prob;
                } else if (dealer_sum > player_sum) {
                    counts.loss += parent_prob;
                } else {
                    CHECK_EQ(dealer_sum, player_sum);
                    counts.draw += parent_prob;
                }
            } else {
                // Dealer will hit.
                for (int card = ACE; card <= TEN; card++) {
                    double prob = parent_prob * card_chance(card);
                    // Use a dummy BlackjackState object, treating the dealer as the player.
                    BlackjackState after_hit = calculate_next_state({dealer_sum, dealer_usable_ace,
                                                                     0}, card);
                    int sum = after_hit.player_sum;
                    bool usable_ace = after_hit.usable_ace;
                    if (sum > MAX_SUM) {
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
        if (env_.is_end_state(from_state)) {
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
        if (from_state.usable_ace) {
            if (from_state.player_sum <= 17) {
                ans = h;
            } else if (from_state.player_sum >= 19) {
                ans = s;
            } else {
                CHECK_EQ(from_state.player_sum, 18);
                if (BlackjackEnvironment::card_value(from_state.dealer_card) >= 9) {
                    ans = h;
                } else {
                    ans = s;
                }
            }
        } else {
            if (from_state.player_sum >= 17) {
                ans = s;
            } else if (BlackjackEnvironment::card_value(from_state.dealer_card) >= 7) {
                ans = h;
            } else if (from_state.player_sum >= 13) {
                ans = s;
            } else if (from_state.player_sum == 12 and
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

} // namespace suttonbarto
} // namespace test
} // namespace rl
