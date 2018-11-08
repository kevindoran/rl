#pragma once

#include <random>
#include <gsl/gsl>

namespace rl {
namespace util {
namespace random {

/**
 * \returns the random number generator being used by all methods in rl::util::random.
 */
std::mt19937& generator();

/**
 * Recreate a mt19937 generator with the given seed.
 */
void reseed_generator(uint seed);

// For ints, longs etc.
template<typename NUM>
std::enable_if_t<std::is_integral<NUM>::value, NUM>
random_in_range(NUM from_inclusive, NUM to_exclusive) {
    Expects(from_inclusive < to_exclusive);
    std::uniform_int_distribution<NUM> dist(from_inclusive, to_exclusive - 1);
    NUM ans = dist(generator());
    Ensures(ans >= from_inclusive);
    Ensures(ans < to_exclusive);
    return ans;
}

// For floats, doubles, etc.
template<typename NUM>
std::enable_if_t<std::is_floating_point<NUM>::value, NUM>
random_in_range(NUM from_inclusive, NUM to_exclusive) {
    Expects(from_inclusive < to_exclusive);
    std::uniform_real_distribution<NUM> dist(from_inclusive, to_exclusive);
    NUM ans = dist(generator());
    Ensures(ans >= from_inclusive);
    Ensures(ans < to_exclusive);
    return ans;
}

} // namespace random
} // namespace util
} // namespace rl

