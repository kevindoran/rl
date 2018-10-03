#ifndef REINFORCEMENT_STOCHASTICPOLICY_H
#define REINFORCEMENT_STOCHASTICPOLICY_H

#include "core/Policy.h"
#include "DeterministicPolicy.h"

namespace rl {

class StochasticPolicy : public Policy {
public:

    // Our need for this method resulted in a refactor of ActionDistribution that saw the class get
    // a DistributionList member. We created the DistributionList because we couldn't use
    // DistributionTree- the latter is not copyable, and we want a copyable ActionDistribution.
    const Action& next_action(const Environment &e, const State &from_state) const override {
        auto finder = state_to_action_dist_.find(from_state);
        Expects(finder != std::end(state_to_action_dist_));
        return finder->second.random_action();
    }

    ActionDistribution
    possible_actions(const Environment &e, const State &from_state) const override {
        auto finder = state_to_action_dist_.find(from_state);
        // This could be either an Ensures or an Expects.
        if(finder == std::end(state_to_action_dist_)) {
            return ActionDistribution{};
        };
        return finder->second;
    }

    void add_action_for_state(const State& s, const Action& a, long weight) {
        // C++17: emplace unless there is something already there, in which case do nothing.
        auto result = state_to_action_dist_.try_emplace(s, ActionDistribution{});
        // result is a pair: an iterator to the inserted pair, and an was_inserted flag.
        Expects(result.first != std::end(state_to_action_dist_));
        ActionDistribution& action_dist = result.first->second;
        action_dist.add_action(a, weight);
    }

    bool clear_actions_for_state(const State& s) {
        std::size_t remove_count =state_to_action_dist_.erase(s);
        Ensures(remove_count <= 1);
        bool something_removed = static_cast<bool>(remove_count);
        return something_removed;
    }

    using StateToActionDistMap = std::unordered_map<State, ActionDistribution>;
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
        StochasticPolicy out;
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

#endif //REINFORCEMENT_STOCHASTICPOLICY_H
