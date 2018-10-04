#pragma once

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

#include "rl/Environment.h"
#include "rl/DistributionTree.h"


// TODO: there isn't much being used in gsl/gsl. Swap out Expects and Ensures for google logging
// methods.

namespace rl {


/**
 * Order by sequence:
 *      state, action, next_state, reward.
 */
struct cumulative_grouping_less {
    bool operator() (const Transition& a, const Transition& b)
    {
        if(a.state().id() != b.state().id()) {
            return a.state().id() < b.state().id();
        }
        if(a.action().id() != b.action().id()) {
            return a.action().id() < b.action().id();
        }
        if(a.next_state().id() != b.next_state().id()) {
            return a.next_state().id() < b.next_state().id();
        }
        return a.reward().id() < b.reward().id();
    }
};

class MappedEnvironment : public Environment {
public:

    // https://stackoverflow.com/questions/32040426/expose-c-container-iterator-to-user
    // Exposing custom iterator types is an alternative option if we want to hide the underlying
    // container implementation, however, it's doesn't seem simple.
    // Instead we will follow one of the solutions outlined here:
    // https://jonasdevlieghere.com/containers-of-unique-pointers/
    // And using the boost feature:
    // https://stackoverflow.com/questions/25669120/iterating-over-const-t-in-a-stdvectorstdunique-ptrt

    explicit MappedEnvironment() = default;
    // Can't have copy unless we manually treat out vector of unique pointers and our dist tree.
    MappedEnvironment(const MappedEnvironment&) = delete;
    MappedEnvironment& operator=(const MappedEnvironment&) = delete;
    MappedEnvironment(MappedEnvironment&&) = default;
    MappedEnvironment& operator=(MappedEnvironment&&) = default;
    ~MappedEnvironment() = default;



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

    void mark_as_end_state(const State& state) {
        end_states_.insert(state.id());
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
        Expects(t.action().id()     < static_cast<ID>(actions_.size()));
        Expects(t.next_state().id() < static_cast<ID>(states_.size()));
        Expects(t.state().id()      < static_cast<ID>(states_.size()));
        Expects(t.reward().id()     < static_cast<ID>(rewards_.size()));
        // We can't have transitions from the end states.
        Expects(end_states_.find(t.state().id()) == std::end(end_states_));
        // Note: using copy ctr below.
        auto res = transitions_.emplace(t);
        bool was_added = res.second;
        Expects(was_added);
        const Transition& added = *res.first;
        needs_rebuilding_ = true;
        return added;
    }

    const Transition& execute_action(const Action& action) override {
        Expects(!is_in_end_state());
        started_ = true;
        if(needs_rebuilding_) {
            build_distribution_tree();
        }
        DistNode& n{get_dist_node(current_state_, action.id())};
        const Transition& random_transition{*CHECK_NOTNULL(n.random_leaf().data())};
        accumulated_reward_+= random_transition.reward().value();
        current_state_= random_transition.next_state().id();
        return random_transition;
    }

    ID state_count() const override {
        return static_cast<ID>(states_.size());
    }

    const State& current_state() const {
        return *states_[current_state_];
    }

    State& current_state() {
        return const_cast<State&>(static_cast<const MappedEnvironment*>(this)->current_state());
    }

    const State& state(ID id) const {
        return *states_[id];
    }

    State& state(ID id) {
        return const_cast<State&>(static_cast<const MappedEnvironment*>(this)->state(id));
    }


    const Action& action(ID id) const {
        return *actions_[id];
    }

    Action& action(ID id) {
        return const_cast<Action&>(static_cast<const MappedEnvironment*>(this)->action(id));
    }

    const Reward& reward(ID id) const {
        return *rewards_[id];
    }

    Reward& reward(ID id) {
        return const_cast<Reward&>(static_cast<const MappedEnvironment*>(this)->reward(id));
    }

    void set_all_rewards_to(double value) {
        for(auto& p_reward : rewards_) {
            p_reward->set_value(value);
        }
    }

    double accumulated_reward() {
        return accumulated_reward_;
    }

    bool is_in_end_state() {
        return end_states_.find(current_state_) != std::end(end_states_);
    }

    void set_start_state(const State& state) {
        start_state_ = state.id();
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

    StateIterator states_begin() const {
        return util::DereferenceIterator(states_.cbegin());
    }

    StateIterator states_end() const {
        return util::DereferenceIterator(states_.cend());
    }

    States states() const {
        return States(states_begin(), states_end());
    }

    ActionIterator actions_begin() const {
        return util::DereferenceIterator(actions_.cbegin());
    }

    ActionIterator actions_end() const {
        return util::DereferenceIterator(actions_.cend());
    }

    Actions actions() const {
        return Actions(actions_begin(), actions_end());
    }

    RewardIterator rewards_begin() const {
        return util::DereferenceIterator(rewards_.cbegin());
    }

    RewardIterator rewards_end() const {
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

    TransitionDistribution transition_list(const State& from_state, const Action& action) const {
        TransitionDistribution ans{};
        Expects(dist_tree_.root_node().has_child_with_id(from_state.id()));
        if(is_end_state(from_state)) {
            return ans;
        }
        const DistNode& state_node = dist_tree_.root_node().child_with_id(from_state.id());
        Expects(state_node.has_child_with_id(action.id()));
        const DistNode& action_node = state_node.child_with_id(action.id());
        auto append_fctn =
             [&ans](const DistNode& node) {
                 if(node.child_count()) {
                     return;
                 }
                 const Transition& t = *CHECK_NOTNULL(node.data());
                 ans.transitions.emplace_back(t);
             };
        dist_tree_.dfs(append_fctn, action_node);
        ans.total_weight = action_node.weight();
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

// Following the examples here:
// https://en.cppreference.com/w/cpp/utility/hash
// Place the hash struct in the "std" namespace so that it is automatically chosen when creating
// maps.
namespace std {

template<> struct hash<rl::State>
{
    // Mark them as inline and we can keep them in the header.
    inline std::size_t operator()(const rl::State& s) const {
        return s.id();
    }
};

template<> struct hash<rl::Action>
{
    inline std::size_t operator()(const rl::Action& a) const {
        return a.id();
    }
};

template<> struct hash<rl::Reward>
{
    inline std::size_t operator()(const rl::Reward& r) const {
        return r.id();
    }
};

} // namespace std
