#ifndef REINFORCEMENT_DETERMINISTICPOLICY_H
#define REINFORCEMENT_DETERMINISTICPOLICY_H

#include <gsl/gsl>
#include <glog/logging.h>

#include "core/Policy.h"

namespace rl {

class DeterministicPolicy : public Policy {

public:
    const Action& next_action(const Environment &e) override {
        auto it = state_to_action_.find(e.current_state());
        Ensures(it != std::end(state_to_action_));
        return it->get();
    }

    void set_action_for_state(const State& s, const Action& a) {
        auto res = state_to_action_.emplace(s, a);
        bool inserted = res.second;
        if(inserted) {
            // log something.
        }
    }

private:
    // Options for this map:
    //    1. ID->ID.
    //    2. State->Action
    //    3. State&->Action&
    //
    //    Benefits of #1 & #2: we are not tying this policy object to a specific environment object.
    //    #3 will tie the policy object to a specific environment.
    //    Benefits of #1 & #3: we only store a single 32 or 64 bit value.
    //    Beneifts of #2: we get very nice debugging capabilities.
    //    For the moment, #2 will be used, and we may switch to #1 if we want better performance.
    //    #3 shouldn't be used, as we would like a single policy to work on multiple Environments,
    //    and we do not wish to restrict the implementation of Environment such that all
    //    Environments use the same state, action and policy objects.
    using StateToActionMap = std::unordered_map<State, Action>;
    StateToActionMap state_to_action_{};
};

} // namespace rl

#endif //REINFORCEMENT_DETERMINISTICPOLICY_H
