#include "gtest/gtest.h"
#include "gsl/gsl_cdf.h"

#include "rl/QeGreedyPolicy.h"
#include "rl/ActionValueFunction.h"
#include "rl/Environment.h"
#include "rl/impl/Environment.h"

namespace {
/**
 * An environment with 1 state and 6 actions (2 of which are never allowed).
 */
class MiniEnv : public rl::impl::Environment {
public:
    static constexpr int allowed_action_count = 4;
    static constexpr int forbidden_action_count = 2;

public:
    MiniEnv() {
        add_state("The only state");
        for(int i = 0; i < allowed_action_count; i++) {
            std::stringstream ss;
            ss << "Allowed action " << i;
            add_action(ss.str());
        }
        for(int i = 0; i < forbidden_action_count; i++) {
            std::stringstream ss;
            ss << "Forbidden action 1" << i;
            add_action(ss.str());
        }
    }

    bool is_action_allowed(const rl::State& from_state, const rl::Action& a) const override {
        bool allowed = a.id() < allowed_action_count;
        return allowed;
    }

    rl::Response next_state(const rl::State& from_state, const rl::Action& action) const override {
        throw std::runtime_error("This method isn't needed for the test.");
    }

    rl::ResponseDistribution
    transition_list(const rl::State& from_state, const rl::Action& action) const override {
        throw std::runtime_error("This method isn't needed for the test.");
    }

    const rl::State& only_state() const {
        return state(0);
    }
};

} // namespace

/**
 * Tests that the greedy policy chooses actions according to it's epsilon.
 */
TEST(QeGreedyPolicy, action_distribution) {
    // Create our environment, a value function with 1 best action and a greedy policy.
    MiniEnv env;
    rl::ActionValueFunction value_fuction(env.state_count(), env.action_count());
    const rl::Action& best_action = env.action(0);
    value_fuction.set_value(env.only_state(), best_action, 10);
    rl::QeGreedyPolicy greedy_policy(value_fuction);
    const double e = 0.3;
    const double random_choice_chance = e / env.allowed_action_count;
    greedy_policy.set_e(e);
    const int iterations = 10000;
    const double significance_level = 0.90;
    rl::util::random::reseed_generator(1);

    // Test
    // a) Tally the policy's actions.
    std::vector<int> action_counts(env.action_count());
    for(int i= 0; i < iterations; i++) {
        const rl::Action& action = greedy_policy.next_action(env, env.only_state());
        action_counts[action.id()]++;
    }
    // b) Check for consistency with epsilon value, via Chi-Squared Test.
    double X2 = 0;
    for(const rl::Action& a : env.actions()) {
        // Action probabilities should be:
        // Not-allowed actions:       0
        // Allowed, but not optimal:  allowed_count/e
        // Best action:              (1-e) + allowed_count/e
        double expected_factor = 0;
        if(env.is_action_allowed(env.only_state(), a)) {
            expected_factor = a == best_action ? (1 - e) + random_choice_chance
                                               : random_choice_chance;
        }
        double expected = expected_factor * iterations;
        double observed = action_counts.at(a.id());
        if(expected == 0) {
            ASSERT_EQ(0, observed);
        } else {
            X2 += std::pow(expected - observed, 2) / expected;
        }
    }
    const int degrees_of_freedom = env.allowed_action_count - 1;
    const double p_value = 1 - gsl_cdf_chisq_P(X2, degrees_of_freedom);
    const double cut_off = 1 - significance_level;
    ASSERT_GT(p_value, cut_off);
}