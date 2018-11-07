#include "rl/Policy.h"

namespace rl {

double error_as_factor(double prev, double updated) {
    double abs_error = std::abs(updated - prev);
    if(abs_error == 0) {
        return 0;
    }
    // We will be conservative about assuming which is more accurate, prev and updated, and
    // choose the smaller one, which will produce the largest error.
    double error_as_factor = 1;
    double denom = std::min(std::abs(prev), std::abs(updated));
    if(denom != 0) {
        error_as_factor = abs_error / denom;
    }
    return error_as_factor;
}

int compare(double val1, double val2, double error_factor) {
    if(error_as_factor(val1, val2) <= error_factor) {
        return 0;
    }
    return (val1 > val2) ? 1 : -1;
}

bool greater_than(double val1, double val2, double by_at_least) {
    return compare(val1, val2, by_at_least) == 1;
}

} // namespace rl
