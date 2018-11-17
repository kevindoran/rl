#pragma once

#include "WindyGridWorld.h"

namespace rl {
namespace test {
namespace suttonbarto {

// Example6_5 doesn't extend TestEnvironment as the example doesn't provide the full optimal
// policy (only the path taken by the optimal policy).
/**
 * Refer to p130 of Sutton & Barto 2018.
 */
class Example6_5 {
public:

    static const std::vector<grid::Position> OPTIMAL_ROUTE;
};

} // namespace suttonbarto
} // namespace test
} // namespace rl
