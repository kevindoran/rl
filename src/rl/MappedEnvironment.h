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
#include "rl/impl/Environment.h"
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

class MappedEnvironment : public impl::Environment {
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

    void set_all_rewards_to(double value) {
        for(auto& p_reward : rewards_) {
            p_reward->set_value(value);
        }
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

    ResponseDistribution transition_list(const State& from_state, const Action& action) const {
        ResponseDistribution ans{};
        // The following will fail if the distribution tree hasn't been built.
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
                 ans.add_response(Response::from_transition(t));
             };
        dist_tree_.dfs(append_fctn, action_node);
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

private:
    DistTree dist_tree_;
    bool needs_rebuilding_ = false;

    std::set<Transition, cumulative_grouping_less> transitions_{};
};

} // namespace rl

