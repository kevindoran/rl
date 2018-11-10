#include "MCEvaluator3.h"

namespace rl {

void MCEvaluator3::initialize(const Environment& env, const Policy& policy) {
    impl::PolicyEvaluator::initialize(env, policy);
    // note: these assignments might be switched to heap construction eventually.
    value_function_ = ActionValueFunction(env.state_count(), env.action_count());
    deltas = StateActionMap<double>(env);
    cumulative_sampling_ratios = StateActionMap<double>(env);
    long initial_count = 0;
    // End states wont be visited and they should not be considered when calculating the min visit.
    long end_state_initial_count = std::numeric_limits<long>::max();
    visit_counts = StateActionMap<long>(env, initial_count, end_state_initial_count);
}

bool MCEvaluator3::finished() const {
    return most_recent_delta_ < delta_threshold_ and min_visit > MIN_VISIT;
}

void MCEvaluator3::step() {
    const Environment& env = *CHECK_NOTNULL(env_);
    Trace trace = run_trial(env, behaviour_policy);
    update_action_value_fctn(trace);
    most_recent_delta_ = *std::max_element(std::begin(deltas.data()), std::end(deltas.data()));
    min_visit = *std::min_element(std::begin(visit_counts.data()), std::end(visit_counts.data()));
    steps_++;
}

void MCEvaluator3::update_action_value_fctn(const Trace& trace) {
    const Environment& env = *CHECK_NOTNULL(env_);
    const Policy& policy = *CHECK_NOTNULL(policy_);
    double retrn = 0;
    double sampling_ratio = 1.0;
    retrn += trace.back().reward;
    for (auto it = std::next(std::crbegin(trace)); it != std::crend(trace); ++it) {
        const TimeStep& ts = *it;
        const Action& action = *CHECK_NOTNULL(ts.action);
        // If the target policy could never take this route, exit.
        double updated_cumulative_weight = cumulative_sampling_ratios.data(ts.state, action)
                                           + sampling_ratio;
        double current_val = value_function_.value(ts.state, action);
        double updated_val = current_val + sampling_ratio / updated_cumulative_weight *
                                           (retrn - current_val);
        // Update data.
        value_function_.set_value(ts.state, action, updated_val);
        visit_counts.data(ts.state, action)++;
        deltas.set(ts.state, action, std::abs(updated_val - current_val));
        cumulative_sampling_ratios.set(ts.state, action, updated_cumulative_weight);
        retrn += ts.reward;
        // note: The sampling ratio is updated _after_ updating the value function. This is done
        // so that we still get estimates for every state-action pair even if the target policy
        // would never take such an action in a given state. By doing this we are able to answer:
        // "If action a is taken in state s then target policy is followed, what is the return?"
        double behaviour_action_prob =
                behaviour_policy.possible_actions(env, ts.state).weight(action);
        double target_action_prob = policy.possible_actions(env, ts.state).weight(action);
        sampling_ratio *= (target_action_prob / behaviour_action_prob);
        if (sampling_ratio == 0.0) {
            break;
        }
    }
}

void MCEvaluator3::set_averaging_mode(MCEvaluator3::AveragingMode mode) {
    averaging_mode_ = mode;
}

const ActionValueFunction& MCEvaluator3::value_function() const {
    return value_function_;
}

} // namespace rl
