#pragma once

#include <random>
#include <gsl/gsl>

namespace rl {
namespace util {


template<typename NUM>
NUM random_in_range(NUM from_inclusive, NUM to_exclusive) {
    // Alternatively, we could define these as static variables in this header and define them in a
    // source file.
    static std::random_device random_device;
    static std::mt19937 random_gen{random_device()};
    Expects(from_inclusive < to_exclusive);
    std::uniform_int_distribution<NUM> dist(from_inclusive, to_exclusive - 1);
    NUM ans = dist(random_gen);
    Ensures(ans >= from_inclusive);
    Ensures(ans < to_exclusive);
    return ans;
}

} // namespace util
} // namespace rl

