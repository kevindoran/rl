#ifndef REINFORCEMENT_ENVIRONMENT_H
#define REINFORCEMENT_ENVIRONMENT_H

#include <cstdlib>
#include <string>
#include <functional>
#include <set>
#include <unordered_set>
#include <gsl/gsl>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <glog/logging.h>

#include "core/DistributionTree.h"
#include "util/DereferenceIterator.h"
#include "util/RangeWrapper.h"


// TODO: there isn't much being used in gsl/gsl. Swap out Expects and Ensures for google logging
// methods.

namespace rl {

/*
 * Following the reasoning here and avoiding unsigned integer types.
 * https://google.github.io/styleguide/cppguide.html#Integer_Types
 */
using ID = int;
using Weight = int;


class State {
public:
    // If you need a copy, do it in the interface.
    State(ID id, std::string name)
    : id_(id), name_(std::move(name))
    {}

    explicit State(ID id)
    : id_(id)
    {}

    const std::string& name() const {return name_;}
    ID id() const {return id_;}

    bool operator==(const State& other) const {
        return id_ == other.id_;
    }

    bool operator!=(const State& other) const {
        return !(*this == other);
    }

private:
    ID id_;
    std::string name_{};
};

class Action {
public:
    Action(ID id, std::string name)
    : id_(id), name_(std::move(name))
    {}

    explicit Action(ID id)
    : id_(id)
    {}

    const std::string& name() const {return name_;}
    ID id() const {return id_;}

    bool operator==(const Action& other) const {
        return id_ == other.id_;
    }

    bool operator!=(const Action& other) const {
        return !(*this == other);
    }

private:
    ID id_;
    std::string name_{};
};

class Reward {
public:
    Reward(ID id, std::string name, double value)
    : id_(id), name_(std::move(name)), value_(value)
    {}

    Reward(const ID id, double value)
    : id_(id), value_(value)
    {}

    ID id() const {return id_;}

    const std::string &name() const {return name_;}

    double value() const {return value_;}

    void set_value(double value) {
        value_ = value;
    }

    bool operator==(const Reward& other) const {
        return id_ == other.id_;
    }

    bool operator!=(const Reward& other) const {
        return !(*this == other);
    }

private:
    ID id_;
    std::string name_{};
    double value_;
};

/**
 * Represents a transition and its probability of occuring.
 *
 * \prob is the probability of moving from \c state to \c next_state and getting reward \c reward
 * when \c action is taken.
 */
class Transition {
public:
    Transition(
        const ID state_,
        const ID next_state_,
        const ID action_,
        const ID reward_,
        const Weight prob_weight=1)
        : state_(state_), next_state_(next_state_), action_(action_), reward_(reward_),
          prob_weight_(prob_weight)
    {}

    ID state() const            {return state_;}
    ID next_state() const       {return next_state_;}
    ID action() const           {return action_;}
    ID reward() const           {return reward_;}
    Weight prob_weight() const  {return prob_weight_;}

    bool operator==(const Transition& other) const {
        return  state_       == other.state_ and
                next_state_  == other.next_state_ and
                action_      == other.action_ and
                reward_      == other.reward_ and
                prob_weight_ == other.prob_weight_;
    }

    bool operator!=(const Transition& other) const {
        return !(*this == other);
    }

private:
    ID state_;
    ID next_state_;
    ID action_;
    ID reward_;
    Weight prob_weight_;
};

/**
 * Order by sequence:
 *      state, action, next_state, reward.
 */
struct cumulative_grouping_less {
    bool operator() (const Transition& a, const Transition& b)
    {
        if(a.state() != b.state()) {
            return a.state() < b.state();
        }
        if(a.action() != b.action()) {
            return a.action() < b.action();
        }
        if(a.next_state() != b.next_state()) {
            return a.next_state() < b.next_state();
        }
        return a.reward() < b.reward();
    }
};



class Environment {
public:

    // https://stackoverflow.com/questions/32040426/expose-c-container-iterator-to-user
    // Exposing custom iterator types is an alternative option if we want to hide the underlying
    // container implementation, however, it's doesn't seem simple.
    // Instead we will follow one of the solutions outlined here:
    // https://jonasdevlieghere.com/containers-of-unique-pointers/
    // And using the boost feature:
    // https://stackoverflow.com/questions/25669120/iterating-over-const-t-in-a-stdvectorstdunique-ptrt

    explicit Environment() = default;
    // Can't have copy unless we manually treat out vector of unique pointers and our dist tree.
    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;
    Environment(Environment&&) = default;
    Environment& operator=(Environment&&) = default;
    ~Environment() = default;

    // Yuck. We still have std::unique_ptr and std::vector in the type. See DereferenceIterator.h
    // for more thoughts.
    using StateIterator =  util::DereferenceIterator<std::vector<std::unique_ptr<State>>::const_iterator>;
    using ActionIterator = util::DereferenceIterator<std::vector<std::unique_ptr<Action>>::const_iterator>;
    using RewardIterator = util::DereferenceIterator<std::vector<std::unique_ptr<Reward>>::const_iterator>;
    using States = util::RangeWrapper<StateIterator>;

    State& add_state(const std::string& name, bool end_state=false) {
        GSL_CONTRACT_CHECK("only max_value(ID) entries are supported.",
                           states_.size() <= std::numeric_limits<ID>::max());
        ID id = (ID) states_.size();
        std::unique_ptr<State>& s = states_.emplace_back(std::make_unique<State>(id, name));
        if(end_state) {
            end_states_.insert(id);
        }
        needs_rebuilding_ = true;
        return *s;
    }

    void mark_as_end_state(ID state_id) {
        end_states_.insert(state_id);
    }

    Action& add_action(const std::string& name) {
        GSL_CONTRACT_CHECK("only max_value(ID) entries are supported.",
                           actions_.size() <= std::numeric_limits<ID>::max());
        needs_rebuilding_ = true;
        return *actions_.emplace_back(std::make_unique<Action>(actions_.size(), name));
    }

    Reward& add_reward(long value, std::string name = {}) {
        GSL_CONTRACT_CHECK("only max_value(ID) entries are supported.",
                           rewards_.size() <= std::numeric_limits<ID>::max());
        needs_rebuilding_ = true;
        return *rewards_.emplace_back(std::make_unique<Reward>(rewards_.size(), name, value));
    }

    const Transition& add_transition(Transition t) {
        GSL_CONTRACT_CHECK("only max_value(ID) entries are supported.",
                           transitions_.size() <= std::numeric_limits<ID>::max());
        // We assert bounds for the array sizes, so the static cast is okay.
        Expects(t.action()     < static_cast<ID>(actions_.size()));
        Expects(t.next_state() < static_cast<ID>(states_.size()));
        Expects(t.state()      < static_cast<ID>(states_.size()));
        Expects(t.reward()     < static_cast<ID>(rewards_.size()));
        // We can't have transitions from the end states.
        Expects(end_states_.find(t.state()) == std::end(end_states_));
        // Note: using copy ctr below.
        auto res = transitions_.emplace(t);
        bool was_added = res.second;
        Expects(was_added);
        const Transition& added = *res.first;
        needs_rebuilding_ = true;
        return added;
    }

    const Transition& execute_action(const Action& action) {
        Expects(!is_in_end_state());
        started_ = true;
        if(needs_rebuilding_) {
            build_distribution_tree();
            needs_rebuilding_ = false;
        }
        DistNode& n{get_dist_node(current_state_, action.id())};
        const Transition& random_transition{*CHECK_NOTNULL(n.random_leaf().data())};
        accumulated_reward_+= reward(random_transition.reward()).value();
        current_state_= random_transition.next_state();
        return random_transition;
    }

    ID state_count() const {
        return static_cast<ID>(states_.size());
    }

    const State& current_state() const {
        return *states_[current_state_];
    }

    State& current_state() {
        return const_cast<State&>(static_cast<const Environment*>(this)->current_state());
    }

    const State& state(ID id) const {
        return *states_[id];
    }

    State& state(ID id) {
        return const_cast<State&>(static_cast<const Environment*>(this)->state(id));
    }

    Action& action(ID id) {
        return *actions_[id];
    }

    Reward& reward(ID id) {
        return *rewards_[id];
    }

    double accumulated_reward() {
        return accumulated_reward_;
    }

    bool is_in_end_state() {
        return end_states_.find(current_state_) != std::end(end_states_);
    }

    void set_start_state(ID state) {
        start_state_ = state;
        if(!started_) {
            current_state_ = start_state_;
        }
    }

    void restart() {
        current_state_ = start_state_;
        accumulated_reward_ = 0;
        started_ = false;
    }

    std::vector<std::reference_wrapper<const State>> end_states() {
        std::vector<std::reference_wrapper<const State>> ans;
        std::transform(end_states_.begin(), end_states_.end(), ans.begin(),
            [this](ID id) {
                return std::cref(state(id));
            });
        return ans;
    }

    bool is_end_state(const State& s) const {
        auto finder = std::find(end_states_.begin(), end_states_.end(), s.id());
        bool is_an_end_state = finder != std::end(end_states_);
        return is_an_end_state;
    }

    StateIterator states_begin() {
        return util::DereferenceIterator(states_.cbegin());
    }

    StateIterator states_end() {
        return util::DereferenceIterator(states_.cend());
    }

    States states() {
        return States(states_begin(), states_end());
    }

    ActionIterator actions_begin() {
        return util::DereferenceIterator(actions_.cbegin());
    }

    ActionIterator actions_end() {
        return util::DereferenceIterator(actions_.cend());
    }

    RewardIterator rewards_begin() {
        return util::DereferenceIterator(rewards_.cbegin());
    }

    RewardIterator rewards_end() {
        return util::DereferenceIterator(rewards_.cend());
    }

    void build_distribution_tree() {
        dist_tree_ = std::move(DistTree());
        DistNode& root = dist_tree_.root_node();

        std::size_t added_count = 0;
        for(const Transition& t : transitions_) {
            if(!root.has_child_with_id(t.state().id())) {
                root.add_child_with_id(t.state().id());
            }
            DistNode& state_node = root.child_with_id(t.state().id());
            if(!state_node.has_child_with_id(t.action().id())) {
                state_node.add_child_with_id(t.action().id());
            }
            DistNode& action_node = state_node.child_with_id(t.action().id());
            if(!action_node.has_child_with_id(t.next_state().id())) {
                action_node.add_child_with_id(t.next_state().id());
            }
            DistNode& next_state_node = action_node.child_with_id(t.next_state().id());
            DistNode& reward = next_state_node.add_child_with_id(t.reward().id(), t.prob_weight(), &t);
            added_count++;
        }
        Ensures(added_count == transitions_.size());
        dist_tree_.update_weights();
        needs_rebuilding_ = false;
    }

    // The methods below provide access to properties of an environment that are not typically
    // available directly.

    // We could make this return copies or references.
    std::vector<std::reference_wrapper<const Transition>> transition_list(const State& from_state,
            const Action& action) {
        std::vector<std::reference_wrapper<const Transition>> ans;
        DistNode& n = dist_tree_.root_node()
                .child_with_id(from_state.id())
                .child_with_id(action.id());
        auto append_fctn =
             [&ans](const DistNode& node) {
                 std::reference_wrapper<const Transition> t =
                         std::cref(*CHECK_NOTNULL(node.data()));
                 ans.emplace_back(t);
             };
        dist_tree_.dfs(append_fctn, n);
        return ans;
    }

    // end

private:
    using DistTree = DistributionTree<const Transition>;
    using DistNode = DistTree::Node;

    DistNode& get_dist_node(ID state) {
        return dist_tree_.root_node().child_with_id(state);
    }

    DistNode& get_dist_node(ID state, ID action) {
        return get_dist_node(state).child_with_id(action);
    }

    DistNode& get_dist_node(ID state, ID action, ID next_state) {
        return get_dist_node(state, action).child_with_id(next_state);
    }

    DistNode& get_dist_node(ID state, ID action, ID next_state, ID reward) {
        return get_dist_node(state, action, next_state).child_with_id(reward);
    }

    /*
    void build_distribution_tree() {
        // Clear the tree so we can call this multiple times.
        dist_tree_ = std::move(DistributionTree<Transition>());
        // std::sort only works with random access iterators.
        // std::sort(std::begin(transitions_), std::end(transitions_), cumulative_grouping_less());
        transitions_.sort(cumulative_grouping_less());
        const auto transition_it{std::begin(transitions_)};
        DistNode& root = dist_tree_.root_node();
        int added_count = 0;
        for(const std::unique_ptr<State>& s : states_) {
            // We should never have a transition that applies to a loop we have already covered.
            Ensures(transition_it->state() >= s->id());
            if(transition_it->state() > s->id()) {
                continue;
            }
            DistNode& state_node = root.add_child();
            Ensures(&state_node == &root.child(s->id()));
            for(const std::unique_ptr<Action>& a : actions_) {
                Ensures(transition_it->action() >= a->id());
                if(transition_it->action() > a->id()) {
                    continue;
                }
                DistNode& action_node = state_node.add_child();
                for(const std::unique_ptr<State>& next_state : states_) {
                    Ensures(transition_it->next_state() >= next_state->id());
                    if(transition_it->next_state() > next_state->id()) {
                        continue;
                    }
                    DistNode& next_state_node = action_node.add_child();
                    for(const std::unique_ptr<Reward>& reward : rewards_) {
                        Ensures(transition_it->reward() >= reward->id());
                        if(transition_it->reward() > reward->id()) {
                            continue;
                        }
                        Transition& matching_transition{*transition_it};
                        std::next(transition_it);
                        // Double check everything is correct:
                        // We should have continued looping if this wasn't a perfect match.
                        Ensures(s->id() == matching_transition.state());
                        Ensures(a->id() == matching_transition.action());
                        Ensures(next_state->id() == matching_transition.next_state());
                        Ensures(reward->id() == matching_transition.reward());

                        next_state_node.add_child(matching_transition.prob_weight(),
                                                  &matching_transition);
                        added_count++;
                    }
                }
            }
        }
        // Make sure we have added all of the transitions to the distribution tree.
        Ensures(transitions_.size() <= std::numeric_limits<ID>::max());
        Ensures(added_count == static_cast<ID>(transitions_.size()));
        dist_tree_.update_weights();
    }
    */

private:
    DistTree dist_tree_;
    ID current_state_ = 0;
    bool started_ = false;
    ID start_state_ = 0;
    double accumulated_reward_ = 0;
    bool needs_rebuilding_ = false;

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
    std::set<Transition, cumulative_grouping_less> transitions_{};

};

} // namespace rl

#endif //REINFORCEMENT_ENVIRONMENT_H
