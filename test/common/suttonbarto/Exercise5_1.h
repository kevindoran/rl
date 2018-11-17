#pragma  once

#include <unordered_map>
#include "TestEnvironment.h"
#include "BlackjackEnvironment.h"

namespace rl {
namespace test {
namespace suttonbarto {

class Exercise5_1 : public TestEnvironment {
public:
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
