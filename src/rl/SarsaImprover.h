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
    const int MIN_VISIT = 100;

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
        long initial_count = 0;
        // End states wont be visited and they should not be considered when calculating the min visit.
        long end_state_initial_count = std::numeric_limits<long>::max();
        StateActionMap<long> visits(env, initial_count, end_state_initial_count);
        StateActionMap<double> deltas(env);
        double min_visit = 0;
        double max_delta = std::numeric_limits<double>::max();
        bool policy_changed = false;
        auto finished = [&]() {
                return max_delta < delta_threshold_ and min_visit > MIN_VISIT and !policy_changed;
            };
        while(!finished()) {
            policy_changed = false;
            // Using exploring starts.
            for(const State& start_state : env.states()) {
                if(env.is_end_state(start_state)) {
                    continue;
                }
                for(const Action& start_action : env.actions()) {
                    if(!env.is_action_allowed(start_state, start_action)) {
                        continue;
                    }
                    Trace trace = run_trial(env, policy, &start_state, &start_action);
                    TimeStep prev_ts = trace.back();
                    for(auto it = std::next(std::crbegin(trace)); it != std::crend(trace); ++it) {
                        const TimeStep& ts = *it;
                        const Action& action = *CHECK_NOTNULL(ts.action);
                        // Update value function.
                        double next_state_val = calculate_state_value(env, value_function,
                                                                      prev_ts.state, policy);
                        double error = prev_ts.reward + next_state_val
                                                      - value_function.value(ts.state, action);
                        const long n = ++visits.data(ts.state, action);
                        double current_val = value_function.value(ts.state, action);
                        double updated_val = current_val + 1.0/n * error;
                        const Action& policy_action_before = policy.next_action(env, ts.state);
                        value_function.set_value(ts.state, action, updated_val);
                        // Update other data.
                        deltas.set(ts.state, action, std::abs(current_val - updated_val));
                        policy_changed = policy_changed or
                                         policy_action_before != policy.next_action(env, ts.state);
                    }
                }
            }
            max_delta = *std::max_element(std::begin(deltas.data()), std::end(deltas.data()));
            min_visit = *std::min_element(std::begin(visits.data()), std::end(visits.data()));
            LOG(INFO) << "max delta: " << max_delta << ", min visit: " << min_visit << std::endl;
        }

        // We can't simply return the greedy policy object, as it depends on the value function
        // variable that is local to this function. And the return signature is for a heap alocated
        // policy anyway. Instead, lets create a new policy from the greedy policy.
        return std::make_unique<StochasticPolicy>(StochasticPolicy::create_from(env, policy));
    }
};

} // namespace rl