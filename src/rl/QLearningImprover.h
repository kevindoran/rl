#pragma once

#include "Policy.h"
#include "Trial.h"
#include "impl/PolicyImprover.h"
#include "QeGreedyPolicy.h"
#include "StochasticPolicy.h"

namespace rl {

class QLearningImprover : public impl::PolicyImprover {
public:
    static constexpr int DEFAULT_ITER_COUNT = 100000;
    static constexpr double DEFAULT_ALPHA = 0.1;
    static constexpr double DEFAULT_GREEDY_E = 0.1;
public:
    /**
     * QLearningImprover doesn't use the input policy parameter.
     */
    std::unique_ptr<Policy> improve(const Environment& env, const Policy&) const override {
        ActionValueFunction value_function(env.state_count(), env.action_count());
        QeGreedyPolicy policy{QeGreedyPolicy::create_pure_greedy_policy(value_function)};
        policy.set_e(greedy_e_);
        for(int i = 0; i < iterations_; i++) {
            Trial trial(env);
            while(!trial.is_finished()) {
                const State& from_state = trial.current_state();
                const Action& action = policy.next_action(env, from_state);
                Response response = trial.execute_action(action);
                // error = reward + (next state's value) - (current value)
                const double next_state_val =
                        value_function.best_action(response.next_state).second;
                const double current_val = value_function.value(from_state, action);
                const double error = response.reward.value() + next_state_val - current_val;
                const double updated_val = current_val + alpha_ * error;
                value_function.set_value(from_state, action, updated_val);
            }
        }
        return std::make_unique<StochasticPolicy>(
                StochasticPolicy::create_from(env, value_function));
    }

    void set_iteration_count(int count) {
        iterations_ = count;
    }

    void set_alpha(double alpha) {
        alpha_ = alpha;
    }

    void set_greedy_e(double e) {
        greedy_e_ = e;
    }

private:
    int iterations_ = DEFAULT_ITER_COUNT;
    double alpha_ = DEFAULT_ALPHA;
    double greedy_e_ = DEFAULT_GREEDY_E;
};

} // namespace rl
