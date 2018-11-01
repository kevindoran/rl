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
        CHECK(state_to_action_dist_.size() < std::numeric_limits<ID>::max());
        CHECK_LT(from_state.id(), static_cast<ID>(state_to_action_dist_.size()));
        const ActionDistribution& dist = state_to_action_dist_.at(from_state.id());
        return dist;
    }

    void add_action_for_state(const State& s, const Action& a, long weight) {
        CHECK_LT(s.id(), state_to_action_dist_.size());
        ActionDistribution& action_dist = state_to_action_dist_.at(s.id());
        action_dist.add_action(a, weight);
    }

    bool clear_actions_for_state(const State& s) {
        CHECK_LT(s.id(), state_to_action_dist_.size());
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
            for(auto entry : other.possible_actions(env, s).weight_map()) {
                const Action& a = *CHECK_NOTNULL(entry.first);
                long weight = entry.second;
                out.add_action_for_state(s, a, weight);
            }
        }
        return out;
    }
};

} // namespace rl
