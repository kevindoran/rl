#pragma once

#include <vector>

#include "rl/MappedEnvironment.h"

namespace rl {

// ValueFunction could be made templated and switch to an array backing.
// This would require either:
//   * Methods that use ValueFunction to become templated.
//   * ValueFunction be given a virtual interface, and the implementation remains templated.
//template<int STATE_COUNT>

/**
 * Represents a state-value function.
 */
class ValueFunction {

public:
    // Core guidelines C21:
    // If you define or delete any default operations, define or delete them all.
    ValueFunction() = delete;
    explicit ValueFunction(ID state_count) : state_values_(state_count, 0) {}
    ValueFunction(const ValueFunction&) = default;
    ValueFunction& operator=(const ValueFunction&) = default;
    ValueFunction(ValueFunction&&) = default;
    ValueFunction& operator=(ValueFunction&&) = default;
    ~ValueFunction() = default;

    double value(const State& state) const {
        Expects(state.id() < static_cast<ID>(state_values_.size()));
        return state_values_[state.id()];
    }

    void set_value(const State& state, double value) {
        Expects(state.id() < static_cast<ID>(state_values_.size()));
        state_values_[state.id()] = value;
    }

private:
    std::vector<double> state_values_;
};

}
