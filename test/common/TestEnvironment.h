#pragma once

#include <memory>
#include <unordered_set>

#include "rl/Environment.h"

namespace rl {
namespace test {

class TestEnvironment {
public:
    // We could also use the following type, but it would require a hash for Action.
    // using OptimalActions = std::unordered_set<std::reference_wrapper<const Action>>;
    using OptimalActions = std::unordered_set<ID>;

    virtual std::string name() const = 0;

    virtual const Environment& env() const = 0;

    virtual double required_discount_rate() const = 0;

    virtual double required_delta_threshold() const = 0;

    virtual OptimalActions optimal_actions(const State& from_state) const = 0;

    // We could also add:
    // optimal_state_value_fuction()
    // optimal_state_action_value_fuction()

    virtual ~TestEnvironment() = default;
};

} // namespace test
} // namespace rl
