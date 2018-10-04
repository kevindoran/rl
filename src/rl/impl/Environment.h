#pragma once

#include <gsl/gsl>
#include <glog/logging.h>
#include <vector>
#include <unordered_set>
#include <memory>

#include "rl/Environment.h"
#include "util/DereferenceIterator.h"
#include "util/RangeWrapper.h"


namespace rl {
namespace impl {
class Environment;
} // namespace impl

// This idea for naming came from Core Guidelines C.129. It seems a
// little hacky though, for example: not being able to define the class within the nested namespace.
// I'll leave it like this for the moment and see what it's like to use.

/**
 * Inheritance base class providing common method implementations for Environment.
 *
 * This class is virtual and can only exist as a base class of an object. Thus, I have marked all
 * constructors as protected.
 */
class impl::Environment : public rl::Environment {

protected:
    Environment() = default;
    // Can't have copy unless we manually treat out vector of unique pointers.
    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;
    // Let's see if we can do move correctly with an inheritance hierachy.
    Environment(Environment&&) = default;
    Environment& operator=(Environment&&) = default;
    // Protected and non-virtual destructor. Following reasoning from Core Guidelines:
    // http://www.modernescpp.com/index.php/c-core-guidelines-destructor-rules
    ~Environment() = default;

public:
    ID state_count() const override {
        return static_cast<ID>(states_.size());
    }

    const State& current_state() const override {
        Ensures(current_state_ < static_cast<ID>(states_.size()));
        return *CHECK_NOTNULL(states_[current_state_]);
    }

    State& current_state() override {
        return const_cast<State&>(static_cast<const impl::Environment*>(this)->current_state());
    }

    const State& state(ID id) const override {
        Expects(id < static_cast<ID>(states_.size()));
        return *CHECK_NOTNULL(states_[id]);
    }

    State& state(ID id) override {
        return const_cast<State&>(static_cast<const impl::Environment*>(this)->state(id));
    }

    const Action& action(ID id) const override {
        Expects(id < static_cast<ID>(actions_.size()));
        return *CHECK_NOTNULL(actions_[id]);
    }

    Action& action(ID id) override {
        return const_cast<Action&>(static_cast<const impl::Environment*>(this)->action(id));
    }

    const Reward& reward(ID id) const override {
        Expects(id < static_cast<ID>(rewards_.size()));
        return *CHECK_NOTNULL(rewards_[id]);
    }

    Reward& reward(ID id) override {
        return const_cast<Reward&>(static_cast<const impl::Environment*>(this)->reward(id));
    }

    double accumulated_reward() override {
        return accumulated_reward_;
    }

    bool is_in_end_state() override {
        return end_states_.find(current_state_) != std::end(end_states_);
    }

    std::vector<std::reference_wrapper<const State>> end_states() override {
        std::vector<std::reference_wrapper<const State>> ans;
        std::transform(end_states_.begin(), end_states_.end(), ans.begin(),
                       [this](ID id) {
                           return std::cref(state(id));
                       });
        return ans;
    }

    bool is_end_state(const State& s) const override {
        auto finder = std::find(end_states_.begin(), end_states_.end(), s.id());
        bool is_an_end_state = finder != std::end(end_states_);
        return is_an_end_state;
    }

    StateIterator states_begin() const override {
        return util::DereferenceIterator(states_.cbegin());
    }

    StateIterator states_end() const override {
        return util::DereferenceIterator(states_.cend());
    }

    States states() const override {
        return States(states_begin(), states_end());
    }

    ActionIterator actions_begin() const override {
        return util::DereferenceIterator(actions_.cbegin());
    }

    ActionIterator actions_end() const override {
        return util::DereferenceIterator(actions_.cend());
    }

    Actions actions() const override {
        return Actions(actions_begin(), actions_end());
    }

    RewardIterator rewards_begin() const override {
        return util::DereferenceIterator(rewards_.cbegin());
    }

    RewardIterator rewards_end() const override {
        return util::DereferenceIterator(rewards_.cend());
    }

    void set_start_state(const State& state) override {
        start_state_ = state.id();
        if(!started_) {
            current_state_ = start_state_;
        }
    }

    void restart() override {
        current_state_ = start_state_;
        accumulated_reward_ = 0;
        started_ = false;
    }

protected:
    ID current_state_ = 0;
    double accumulated_reward_ = 0;
    bool started_ = false;
    ID start_state_ = 0;

    /* Using list vs vector.
     * We wish to be able to hold references/pointers to elements of the below containers. As the
     * locations of vector entries may move, we would need to use std::vector<std::unique_ptr<Type>>
     * to store elements in a vector. To make things a little simplier we could instead use a list.
     * The list guarantees the stable location of its elements on the heap.
     */
    std::vector<std::unique_ptr<State>> states_{};
    std::unordered_set<ID> end_states_{};
    std::vector<std::unique_ptr<Action>> actions_{};
    std::vector<std::unique_ptr<Reward>> rewards_{};
};

} // namespace rl

