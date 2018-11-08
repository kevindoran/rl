#pragma once

#include "rl/Policy.h"
#include "DeterministicPolicy.h"

namespace rl {

class StochasticPolicy : public Policy {
public:

    StochasticPolicy(ID state_count) : state_to_action_dist_(state_count)
    {}

    // Our need for this method resulted in a refactor of ActionDistribution that saw the class get
    // a DistributionList member. We created the DistributionList because we couldn't use
    // DistributionTree- the latter is not copyable, and we want a copyable ActionDistribution.
    const Action& next_action(const Environment &e, const State &from_state) const override {
        return possible_actions(e, from_state).random_action();
    }

    ActionDistribution
    possible_actions(const Environment &e, const State &from_state) const override {
        CHECK_GT(state_to_action_dist_.size(), static_cast<std::size_t>(from_state.id()));
        const ActionDistribution& dist = state_to_action_dist_.at(from_state.id());
        return dist;
    }

    void add_action_for_state(const State& s, const Action& a, Weight weight) {
        CHECK_GT(state_to_action_dist_.size(), static_cast<std::size_t>(s.id()));
        ActionDistribution& action_dist = state_to_action_dist_.at(s.id());
        action_dist.add_action(a, weight);
    }

    bool clear_actions_for_state(const State& s) {
        CHECK_GT(state_to_action_dist_.size(), static_cast<std::size_t>(s.id()));
        ActionDistribution& action_dist = state_to_action_dist_.at(s.id());
        bool existing = !action_dist.empty();
        state_to_action_dist_[s.id()] = ActionDistribution();
        return existing; // i.e. something was reset.
    }

    // We are guaranteed a continuous state ID starting from 0, so we can use a vector as a map.
    using StateToActionDistMap = std::vector<ActionDistribution>;
    StateToActionDistMap state_to_action_dist_{};

    /**
     * Create a \c StochasticPolicy from another policy.
     *
     * Stochastic policies are effectively as general as can be. Thus they can be created from
     * any other policy.
     *
     * Making this a static method for the moment, as I don't want to deal with the construction,
     * copy, move etc for the class hierarchy yet.
     */
    template<typename PolicyInType>
    static StochasticPolicy create_from(const Environment& env, const PolicyInType& other) {
        // note: using unique_ptr is probably better here. I'll leave it as move-return for now,
        // as I'm curious if it will ever become an issue.
        StochasticPolicy out(env.state_count());
        for(const State& s : env.states()) {
            // As the ActionDistribution is returned by value, the full statement
            // other.possible_actions(env, s).weight_map() cannot be placed in the for loop, as
            // ActionDistribution object is destroyed before the loop can begin. This is an easy
            // language trap to fall into. There are suggestions to extend the lifetime of
            // temporaries in such for loop expressions:
            //     http://open-std.org/JTC1/SC22/WG21/docs/cwg_closed.html#900
            ActionDistribution dist = other.possible_actions(env, s);
            for(auto const& entry : dist.weight_map()) {
                const Action& a = *CHECK_NOTNULL(entry.first);
                Weight weight = entry.second;
                out.add_action_for_state(s, a, weight);
            }
        }
        return out;
    }
};

} // namespace rl
