#include "Policy.h"
#include "TDEvaluator.h"
#include <rl/BlendedPolicy.h>

namespace rl {

const ActionValueFunction& TDEvaluator::value_function() const {
    return value_function_;
}

bool TDEvaluator::finished() const {
    return most_recent_delta_ < delta_threshold_ and min_visit > MIN_VISIT;
}

void TDEvaluator::initialize(const Environment& env, const Policy& policy) {
    impl::PolicyEvaluator::initialize(env, policy);
    // note: these assignments might be switched to heap construction eventually.
    value_function_ = ActionValueFunction(env.state_count(), env.action_count());
    deltas = StateActionMap<double>(env);
    long initial_count = 0;
    // End states wont be visited and they should not be considered when calculating the min visit.
    long end_state_initial_count = std::numeric_limits<long>::max();
    visit_counts = StateActionMap<long>(env, initial_count, end_state_initial_count);
}

void TDEvaluator::step() {
    const Environment& env = *CHECK_NOTNULL(env_);
    for (const State& start_state : env.states()) {
        if (env.is_end_state(start_state)) {
            continue;
        }
        for (const Action& start_action : env.actions()) {
            if (!env.is_action_allowed(start_state, start_action)) {
                continue;
            }
            Trace trace = run_trial(env, *CHECK_NOTNULL(policy_), &start_state, &start_action);
            update_value_fctn(trace);
        }
    }
    most_recent_delta_ = *std::max_element(std::begin(deltas.data()), std::end(deltas.data()));
    min_visit = *std::min_element(std::begin(visit_counts.data()), std::end(visit_counts.data()));
    steps_++;
}

void TDEvaluator::update_value_fctn(const Trace& trace) {
    const TimeStep* p_prev_ts = &trace.back();
    for (auto it = std::next(std::crbegin(trace)); it != std::crend(trace); ++it) {
        const TimeStep prev_ts = *CHECK_NOTNULL(p_prev_ts);
        const TimeStep& ts = *it;
        const Action& action = *CHECK_NOTNULL(ts.action);
        // Update value function.
        double current_val = value_function_.value(ts.state, action);
        // This line below distinguishes TD from MC. The next state's state value is being used
        // (like Expected Sarsa) instead of the subsequent state-action pair's state action value.
        double state_val = calculate_state_value(*CHECK_NOTNULL(env_), value_function_,
                                                 prev_ts.state, *CHECK_NOTNULL(policy_));
        double td_error = prev_ts.reward + state_val - value_function_.value(ts.state, action);
        long n = ++visit_counts.data(ts.state, action);
        CHECK_GT(n, 0);
        double updated_val = current_val +  1.0/n * td_error;
        // Update data.
        value_function_.set_value(ts.state, action, updated_val);
        deltas.set(ts.state, action, std::abs(updated_val - current_val));
        p_prev_ts = &ts;
    }
}

} // namespace rl
