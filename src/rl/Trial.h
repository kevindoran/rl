#pragma once

#include "rl/Environment.h"
#include "glog/logging.h"

namespace rl {

class Trial {

public:
    Trial(Environment& env) : env_(env), current_state_(&env.start_state()){}

    // Deleted until needed.
    Trial(const Trial&) = delete;
    Trial(Trial&&) = delete;
    Trial& operator=(const Trial&) = delete;
    Trial& operator=(Trial&&) = delete;

    const State& execute_action(const Action& a) {
        rl::Response response = env_.next_state(*CHECK_NOTNULL(current_state_), a);
        accumulated_reward_ += response.reward.value();
        current_state_ = &response.next_state;
        return *current_state_;
    }

    const State& current_state() const {
        return *current_state_;
    }

    double accumulated_reward() const {
        return accumulated_reward_;
    }

    const Environment& env() const {
        return env_;
    }

private:
    Environment& env_;
    // Using a pointer (instead of reference) so that State object can be kept const and the pointer
    // variable can be still assignable. This is required for the class to allow copy & move.
    const State* current_state_ = nullptr;
    double accumulated_reward_;
};

} // namespace
