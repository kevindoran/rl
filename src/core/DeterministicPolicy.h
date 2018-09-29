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
    // We could just have a map from ID->ID. State->Action, however, allows for easy reporting
    // and debugging. Hmm... but using the references makes everything difficult to copy.
    using StateToActionMap = std::unordered_map<std::reference_wrapper<State>,
                                                std::reference_wrapper<Action>>;
    StateToActionMap state_to_action_{};
};

} // namespace rl

#endif //REINFORCEMENT_DETERMINISTICPOLICY_H
