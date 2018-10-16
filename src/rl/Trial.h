#pragma once

#include "rl/Environment.h"
#include "rl/Policy.h"
#include "glog/logging.h"

namespace rl {

/*
 * A TimeStep represents, for some step tx:
 *   * The reward value obtained at tx (entering the state at tx)
 *   * The state at tx
 *   * The action executed at tx (leaving the state at tx)
 */
struct TimeStep {
    const State& state;
    // Action is nullable, as there is no action taken from an end state.
    const Action* action;
    const double reward;
};
using Trace = std::vector<TimeStep>;

class Trial {

public:
    explicit Trial(const Environment& env) :
        env_(&env), current_state_(&env.start_state()) {}

    Trial(const Environment& env, const State& start_state) :
        env_(&env), current_state_(&start_state) {}

    // Deleted until needed.
    Trial(const Trial&) = delete;
    Trial(Trial&&) = delete;
    Trial& operator=(const Trial&) = delete;
    Trial& operator=(Trial&&) = delete;

    Response execute_action(const Action& a) {
        Response response = env().next_state(current_state(), a);
        accumulated_reward_ += response.reward.value();
        current_state_ = &(response.next_state);
        return response;
    }

    const State& current_state() const {
        return *CHECK_NOTNULL(current_state_);
    }

    double accumulated_reward() const {
        return accumulated_reward_;
    }

    const Environment& env() const {
        return *CHECK_NOTNULL(env_);
    }

    bool is_finished() const {
        return env().is_end_state(current_state());
    }

private:
    // Using a pointer (instead of reference) so that State object can be kept const and the pointer
    // variable can be still assignable. This is required for the class to allow copy & move.
    // Could also use a reference wrapper.
    const Environment* env_;
    const State* current_state_ = nullptr;
    double accumulated_reward_ = 0;
};

// TODO: move to .cc
inline Trace run_trial(
        const Environment& env, const Policy& policy, const State& custom_start_state) {
    Trace trace;
    Trial trial(env, custom_start_state);
    double reward = 0;
    while(!trial.is_finished()) {
        const Action& action = policy.next_action(env, trial.current_state());
        trace.emplace_back(TimeStep{trial.current_state(), &action, reward});
        Response response = trial.execute_action(action);
        reward = response.reward.value();
    }
    // Place the end state in the trace.
    trace.emplace_back(TimeStep{trial.current_state(), nullptr, reward});
    return trace;
}

} // namespace
