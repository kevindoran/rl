#pragma once

#include "rl/Environment.h"
#include <glog/logging.h>

namespace rl {

template<class T>
class StateActionMap {
public:
    using SizeType = long;
    using Container = std::vector<T>;
public:
    // Copying the relevant comment from ActionValueTable:
    // This was changed from deleted -> default as it is convenient for some of the evaluators to
    // store an ActionValueTable by value. It may become useful to revert this change and store
    // via pointers to heap allocated mem. Allowing the default construction allows for a somewhat
    // invalid state to be permitted.
    StateActionMap() = default;

    explicit StateActionMap(const Environment& env) :
            action_count(env.action_count()),
            // This multiplication will be checked for overflow (assuming gcc & -ftrapv is used, and
            // that it covers literal overflow, which I'm not 100% sure that it does).
            data_(static_cast<std::size_t>(env.state_count() * action_count))
    {}

    StateActionMap(const Environment& env, const T& default_val) :
            action_count(env.action_count()),
            // This multiplication will be checked for overflow (assuming gcc & -ftrapv is used, and
            // that it covers literal overflow, which I'm not 100% sure that it does).
            data_(static_cast<std::size_t>(env.state_count() * action_count), default_val)
    {}

    StateActionMap(const Environment& env, const T& default_val, const T& end_state_default_val) :
            StateActionMap(env, default_val)
    {
        for(const State& s : env.end_states()) {
            for(const Action& a : env.actions()) {
                data_[hash(s.id(), a.id())] = end_state_default_val;
            }
        }
    }

    const T& data(const State& s, const Action& a) const {
        std::size_t index = static_cast<std::size_t>(hash(s.id(), a.id()));
        CHECK_GT(data_.size(), index);
        return data_[index];
    }

    T& data(const State& s, const Action& a) {
        return const_cast<T&>(static_cast<const StateActionMap*>(this)->data(s, a));
    }

    // Note: the data is copied.
    void set(const State& s, const Action& a, T data) {
        std::size_t index = static_cast<std::size_t>(hash(s.id(), a.id()));
        CHECK_GT(data_.size(), index);
        data_[index] = data;
    }

    // Being a bit lazy here and exposing the full container.
    const Container& data() const {
        return data_;
    }

private:
    SizeType hash(ID state_id, ID action_id) const {
        CHECK_GT(action_count, action_id);
        return state_id * action_count + action_id;
    }

    /**
     * Useful for debugging.
     */
    std::pair<ID, ID> reverse_hash(SizeType hash) const {
        ID state_id = static_cast<ID>(hash / action_count);
        ID action_id = static_cast<ID>(hash % action_count);
        return {state_id, action_id};
    }

private:
    ID action_count;
    std::vector<T> data_;
};

} // namespace rl
