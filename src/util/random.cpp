#include "random.h"

namespace {
    std::random_device rd{};
    std::mt19937 gen{rd()};
}

namespace rl {
namespace util {
namespace random {

std::mt19937& generator() {
    return gen;
}

void reseed_generator(uint seed) {
    gen = std::mt19937(seed);
}

} // namespace random
} // namespace util
} // namespace rl
