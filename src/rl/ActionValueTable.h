#pragma once

#include "rl/Environment.h"
#include "glog/logging.h"
#include <vector>

namespace rl {

/**
 * Represents an action-value function.
 */
class ActionValueTable {
public:
    // This was changed from deleted -> default as it is convenient for some of the evaluators to
    // store an ActionValueTable by value. It may become useful to revert this change and store
    // via pointers to heap allocated mem. Allowing the default construction allows for a somewhat
    // invalid state to be permitted.
    ActionValueTable() = default;

    ActionValueTable(ID state_count, ID action_count)
    {
        Expects(state_count > 0);
        Expects(action_count > 0);
        for(ID state = 0; state < state_count; state++) {
            values_.emplace_back(std::vector<double>(action_count, 0));
        }
    }

    // Core guidelines C21:
    // If you define or delete any default operations, define or delete them all.
    ActionValueTable(const ActionValueTable&) = default;
    ActionValueTable& operator=(const ActionValueTable&) = default;
    ActionValueTable(ActionValueTable&&) = default;
    ActionValueTable& operator=(ActionValueTable&&) = default;

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

    using ActionValuePair = std::pair<ID, double>;
    ActionValuePair best_action(const State& state) const {
        int max_pos = 0;
        double max_val = std::numeric_limits<double>::lowest();
        CHECK_LT(state.id(), static_cast<ID>(values_.size()));
        const std::vector<double>& action_list = values_.at(state.id());
        for(ID i = 0; i < static_cast<ID>(action_list.size()); i++) {
            if(action_list.at(i) > max_val) {
                max_val = action_list.at(i);
                max_pos = i;
            }
        }
        return std::make_pair(max_pos, max_val);
    }

private:
    using StateActionTable = std::vector<std::vector<double>>;
    StateActionTable values_{};
};

} // namespace rl