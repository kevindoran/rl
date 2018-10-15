#pragma once

#include "rl/Environment.h"
#include "glog/logging.h"

namespace rl {

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

} // namespace
