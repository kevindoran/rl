#pragma once

#include "rl/impl/Environment.h"

namespace rl {
namespace test {
namespace suttonbarto {

/**
 * An environment consisting of 1000 states (and 2 end states) in a line with no actions.
 *
 * From Sutton & Barto 2108 p203.
 */
class RandomWalk1000: public impl::Environment {
public:
    static constexpr int INNER_STATE_COUNT = 1000;
    static constexpr int JUMP = 100;
    static constexpr int START_STATE = 500;
    static constexpr double LEFT_REWARD = -1;
    static constexpr double RIGHT_REWARD = 1;
public:
    RandomWalk1000() {
        // Add states.
        add_end_state("left terminal");
        for(int i = 1; i <= INNER_STATE_COUNT; i++) {
            add_state(std::to_string(i));
        }
        add_end_state("right terminal");
        // Although the Random Walk environment doesn't have any actions, the API design we have
        // currently makes this a little tricky. So we will create a single dummy action.
        add_action("dummy");
        set_start_state(state(START_STATE));
    }
    bool is_action_allowed(const State& from_state, const Action& a) const override {
        // Left and right actions are always allowed.
        return true;
    }

    Response next_state(const State& from_state, const Action& action) const override {
        // We could be lazy and just use:
        // ResponseDistribution dist = transition_list(from_state, action);
        // Response r = dist.random();
        // But our trials will be much slower due to the 200x loops done in that method.
        const int random = util::random::random_in_range<int>(1, JUMP + 1);
        const int move_right = util::random::random_in_range<int>(0, 2);
        const int next_state_id = move_right ? from_state.id() + random : from_state.id() - random;
        const int rounded_id = std::max(0, std::min(state_count()-1, next_state_id));
        // TODO: remove Environmnt's need to return the probability here.
        Weight weight = 1 + std::abs(next_state_id - rounded_id);
        const State& next_state = state(rounded_id);
        double reward = 0;
        if(next_state == left_end()) {
            reward = LEFT_REWARD;
        } else if(next_state == right_end()) {
            reward = RIGHT_REWARD;
        }
        return Response{next_state, Reward(reward), weight};
    }

    /**
     * For the random walk, there is no action taken; all transitions are dependent on the
     * environment only.
     */
    ResponseDistribution transition_list(const State& from_state, const Action&) const override {
        ResponseDistribution res;
        int j = 1;
        for(;j <= JUMP and (from_state.id() - j > 0); j++) {
            res.add_response(Response{state(from_state.id() - j), Reward(0), 1});
        }
        if(j <= JUMP) {
            Weight w = 1 + JUMP - j;
            res.add_response(Response{left_end(), Reward(LEFT_REWARD), w});
        }
        j = 1;
        for(; j <= JUMP and (from_state.id() + j < state_count() - 1); j++) {
            res.add_response(Response{state(from_state.id() + j), Reward(0), 1});
        }
        if(j <= JUMP) {
            Weight w = 1 + JUMP - j;
            res.add_response(Response{right_end(), Reward(RIGHT_REWARD), w});
        }
        return res;
    }

    const State& left_end() const {
        return state(0);
    }

    const State& right_end() const {
        return *states_.back();
    }
};

} // namespace suttonbarto
} // namespace test
} // namespace rl
