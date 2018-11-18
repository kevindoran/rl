#pragma once

#include <vector>

#include "rl/MappedEnvironment.h"

namespace rl {

// ValueTable could be made templated and switch to an array backing.
// This would require either:
//   * Methods that use ValueTable to become templated.
//   * ValueTable be given a virtual interface, and the implementation remains templated.
//template<int STATE_COUNT>

/**
 * Represents a state-value function.
 */
class ValueTable {

public:
    // Core guidelines C21:
    // If you define or delete any default operations, define or delete them all.

    // The default constructor was changed from deleted -> default as it is convenient for some of
    // the evaluators to store an ActionValueTable by value. It may become useful to revert this
    // change and store via pointers to heap allocated mem. Allowing the default construction allows
    // for a somewhat invalid state to be permitted.
    ValueTable() = default;
    explicit ValueTable(ID state_count) : state_values_(state_count, 0) {}
    ValueTable(const ValueTable&) = default;
    ValueTable& operator=(const ValueTable&) = default;
    ValueTable(ValueTable&&) = default;
    ValueTable& operator=(ValueTable&&) = default;
    ~ValueTable() = default;

    double value(const State& state) const {
        Expects(state.id() < static_cast<ID>(state_values_.size()));
        return state_values_[state.id()];
    }

    void set_value(const State& state, double value) {
        Expects(state.id() < static_cast<ID>(state_values_.size()));
        state_values_[state.id()] = value;
    }

private:
    std::vector<double> state_values_{};
};

} // namespace rl
