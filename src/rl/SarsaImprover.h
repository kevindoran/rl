#pragma once

#include "impl/PolicyImprover.h"
#include "StateActionMap.h"
#include "Trial.h"
#include "QeGreedyPolicy.h"
#include "StochasticPolicy.h"

namespace rl {

/**
 * This improver doesn't use an evaluator. This was done partly as an experiment and partly because
 * it allows the improver to easily determine if the policy has updated between iterations. This
 * check is used to determine when to end the policy improvement.
 *
 * Exploring starts is used instead of e-greedy. e-greedy can't be used until Environment is
 * updated to list all valid start states.
 */
class SarsaImprover : public impl::PolicyImprover {
public:
    static constexpr int DEFAULT_ITER_COUNT = 100000;
    static constexpr double DEFAULT_ALPHA = 0.1;
    static constexpr double DEFAULT_GREEDY_E = 0.1;
public:
    /**
     *
     * unused param policy Sarsa doesn't use an input policy as a starting point. The input policy
     *                     is ignored.
     * \return
     */
    std::unique_ptr<Policy> improve(const Environment& env, const Policy&) const override {
        // TODO: improvers should be broken into initialize(), step() & finished() call also.
        ActionValueFunction value_function =
                ActionValueFunction(env.state_count(), env.action_count());
        // How does the following work? no move or copy ctr...
        QeGreedyPolicy policy{QeGreedyPolicy::create_pure_greedy_policy(value_function)};
        policy.set_e(greedy_e);
        for(int i = 0; i < iterations; i++) {
            Trial trial(env);
            while(!trial.is_finished()) {
                const State& from_state = trial.current_state();
                const Action& action = policy.next_action(env, trial.current_state());
                Response response = trial.execute_action(action);
                double next_state_val = calculate_state_value(env, value_function,
                        response.next_state, policy);
                double current_val = value_function.value(from_state, action);
                double error = response.reward.value() + next_state_val - current_val;
                double updated_val = current_val + alpha * error;
                value_function.set_value(from_state, action, updated_val);
            }
        }

        // We can't simply return the greedy policy object, as it depends on the value function
        // variable that is local to this function. And the return signature is for a heap alocated
        // policy anyway. Instead, lets create a new policy from the greedy policy.
        // Don't create the answer from the eGreedyPolicy, create it from the action value function.
        // return std::make_unique<StochasticPolicy>(StochasticPolicy::create_from(env, policy));
        return std::make_unique<StochasticPolicy>(
                StochasticPolicy::create_from(env, value_function));
    }

    void set_iteration_count(int count) {
        iterations = count;
    }

    void set_greedy_e(double e) {
        greedy_e = e;
    }

private:
    int iterations = DEFAULT_ITER_COUNT;
    double alpha = DEFAULT_ALPHA;
    double greedy_e = DEFAULT_GREEDY_E;
};

} // namespace rl