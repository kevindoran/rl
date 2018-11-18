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

double calculate_state_value(const Environment& env, const ActionValueTable& value_function,
                             const State& state, const Policy& policy) {
    if(env.is_end_state(state)) {
        return 0;
    }
    double state_val = 0;
    Policy::ActionDistribution action_dist = policy.possible_actions(env, state);
    for(const Action& action : env.actions()) {
        // note: adding this check, as it isn't fully described whether it is the policy's
        // responsibility to always return 0 for any action executed from an end state. It seems
        // a bit of a burden for this to be enforced. For example, a simple random policy would
        // need the check for end state.
        if(!env.is_action_allowed(state, action)) {
            continue;
        }
        double sum_part = action_dist.probability(action) * value_function.value(state, action);
        state_val += sum_part;
    }
    return state_val;
}

} // namespace rl
