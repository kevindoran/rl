#ifndef REINFORCEMENT_VALUEFUNCTION_H
#define REINFORCEMENT_VALUEFUNCTION_H

#include <vector>

#include "core/Environment.h"

namespace rl {

// ValueFunction could be made templated and switch to an array backing.
// This would require either:
//   * Methods that use ValueFunction to become templated.
//   * ValueFunction be given a virtual interface, and the implementation remains templated.
//template<int STATE_COUNT>

class ValueFunction {

public:
    // Core guidelines C21:
    // If you define or delete any default operations, define or delete them all.
    ValueFunction() = delete;
    explicit ValueFunction(ID state_count) : state_values_(state_count) {}
    ValueFunction(const ValueFunction&) = default;
    ValueFunction& operator=(const ValueFunction&) = default;
    ValueFunction(ValueFunction&&) = default;
    ValueFunction& operator=(ValueFunction&&) = default;
    ~ValueFunction() = default;

    double value(ID state_id) {
        Expects(state_id < static_cast<ID>(state_values_.size()));
        return state_values_[state_id];
    }

    void set_value(ID state_id, double value) {
        Expects(state_id < static_cast<ID>(state_values_.size()));
        state_values_[state_id] = value;
    }

private:
    std::vector<double> state_values_;
};

}

#endif //REINFORCEMENT_VALUEFUNCTION_H
