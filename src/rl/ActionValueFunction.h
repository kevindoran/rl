#pragma once

#include "rl/Environment.h"
#include <vector>

namespace rl {

/**
 * Represents an action-value function.
 */
class ActionValueFunction {
public:
    ActionValueFunction() = delete;

    ActionValueFunction(ID state_count, ID action_count)
    {
        Expects(state_count > 0);
        Expects(action_count > 0);
        for(ID state = 0; state < state_count; state++) {
            values_.emplace_back(std::vector<double>(action_count, 0));
        }
    }

    // Core guidelines C21:
    // If you define or delete any default operations, define or delete them all.
    ActionValueFunction(const ActionValueFunction&) = default;
    ActionValueFunction& operator=(const ActionValueFunction&) = default;
    ActionValueFunction(ActionValueFunction&&) = default;
    ActionValueFunction& operator=(ActionValueFunction&&) = default;

    // note: how should be behave when a given state-action pair isn't valid?
    // For the moment, this issue is offloaded onto the client.
    // Another option is to use a map as a backing and the absense of a value can be represented.
    // Issues with using the map:
    //   * We can't tell the difference between an invalid state-action pair and the absense of a
    //     value for a valid state-action pair (due to a client not setting a value for alll valid
    //     action-value pairs).
    //   * We would require the client to manually set the value of all end states to zero.
    //     Currently, some of the clients of ValueFunction rely on a zero default.
    double value(const State& state, const Action& action) const {
        Expects(state.id() < static_cast<ID>(values_.size()));
        Expects(action.id() < static_cast<ID>(values_.front().size()));
        return values_[state.id()][action.id()];
    }

    void set_value(const State& state, const Action& action, double value) {
        Expects(state.id() < static_cast<ID>(values_.size()));
        Expects(action.id() < static_cast<ID>(values_.front().size()));
        values_[state.id()][action.id()] = value;
    }

private:
    using StateActionTable = std::vector<std::vector<double>>;
    StateActionTable values_{};
};

} // namespace rl