#include "MCEvaluator3.h"
#include "RandomPolicy.h"
#include <rl/BlendedPolicy.h>

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
    double blend = 0.5;
    p_behaviour_policy = std::make_unique<BlendedPolicy>(&policy, &random_policy, blend);
}

bool MCEvaluator3::finished() const {
    return most_recent_delta_ < delta_threshold_ and min_visit > MIN_VISIT;
}

void MCEvaluator3::step() {
    const Environment& env = *CHECK_NOTNULL(env_);
    // Breaking from Sutton & Barto, I'm using exploring starts for the off-policy importance
    // sampling so that a full evaluation function can be obtained.
    for(const State& start_state : env.states()) {
        if(env.is_end_state(start_state)) {
            continue;
        }
        for(const Action& start_action : env.actions()) {
            if(!env.is_action_allowed(start_state, start_action)) {
                continue;
            }
            // For the (start_state, start_action) pair that has been visited least:
            // Keep trying this state-action pair as start states until the target policy has a
            // non-zero chance of carrying out the _full_ trial. When this happens, the value
            // function for our (start_state, start_action) pair will be updated.
            // This inner loop hopes to reduce the time it takes for the min delta to fall bellow
            // the threshold.
            bool least_visited = (visit_counts.data(start_state, start_action) == min_visit);
            long visit_count_before = visit_counts.data(start_state, start_action);
            auto finished = [&]() {
                bool fin = !least_visited or
                        (least_visited &&
                         visit_count_before != visit_counts.data(start_state, start_action));
                return fin;
            };
            // Loop until we get 1 visit for the (start_state, start_action) pair.
            while(!finished()) {
                Trace trace = run_trial(env, *p_behaviour_policy, &start_state, &start_action);
                update_action_value_fctn(trace);
            }
        }
    }
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
        // Update value function.
        double updated_cumulative_weight = cumulative_sampling_ratios.data(ts.state, action)
                                           + sampling_ratio;
        double current_val = value_function_.value(ts.state, action);
        double updated_val = current_val + sampling_ratio / updated_cumulative_weight *
                                           (retrn - current_val);
        // Update other data.
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
                p_behaviour_policy->possible_actions(env, ts.state).probability(action);
        double target_action_prob = policy.possible_actions(env, ts.state).probability(action);
        CHECK_GT(behaviour_action_prob, 0.0);
        sampling_ratio *= (target_action_prob / behaviour_action_prob);
        // If the target policy could never take this route, exit.
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
