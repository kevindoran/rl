#pragma once

#include "TestEnvironment.h"
#include "CliffWorld.h"

namespace rl {
namespace test {
namespace suttonbarto {

/**
 * Example 6.6: Cliff Walking.
 *
 * Refer to p132 of Sutton & Barto 2018.
 *
 */
class Example6_6 : public TestEnvironment {
public:

    static const std::vector<grid::Position> SAFE_ROUTE;
    static const std::vector<grid::Position> OPTIMAL_ROUTE;

    std::string name() const override {
        return "Example 6.6: Cliff Walking";
    }

    const CliffWorld& env() const override {
        return env_;
    }

    double required_discount_rate() const override {
        return 1.0;
    }

    double required_delta_threshold() const override {
        return 0.001;
    }

    OptimalActions optimal_actions(const State& from_state) const override {
        OptimalActions ans;
        std::transform(
                std::begin(optimal_actions_[from_state.id()]),
                std::end(optimal_actions_[from_state.id()]),
                std::inserter(ans, std::end(ans)),
                [this](int dir) {
                    return env_.dir_to_action(grid::directions[dir]).id();
                }
        );
        return ans;
    }

private:
    CliffWorld env_;
    static const std::unordered_set<int> optimal_actions_[CliffWorld::HEIGHT * CliffWorld::WIDTH];
};

} // namespace suttonbarto
} // namespace test
} // namespace rl
