#pragma once

#include "impl/PolicyEvaluator.h"
#include "StateActionMap.h"
#include "Trial.h"
#include "RandomPolicy.h"

namespace rl {

/**
 * Monte Carlo off-policy prediction via importance sampling.
 *
 * Trials using a policy that has full coverage of the policy space (the behaviour policy) are used
 * to calculate the state-action value function for a target policy. The target policy in this case
 * is a greedy policy- greedy with respect to the value function.
 *
 * I've given up on trying to encode the details of the algorithms into the class names.
 */
class MCEvaluator3 : public ActionBasedEvaluator,
                     public impl::PolicyEvaluator {
public:
    // TODO: these options aren't used yet.
    enum class AveragingMode {STANDARD, WEIGHTED};
    static const int MIN_VISIT = 100;

public:
    void set_averaging_mode(AveragingMode mode);
    void initialize(const Environment& env, const Policy& policy) override;
    void step() override;
    bool finished() const override;
    const ActionValueTable& value_function() const override;

private:
    void update_action_value_fctn(const Trace& trace);

private:
    AveragingMode averaging_mode_ = AveragingMode::WEIGHTED;
    std::unique_ptr<Policy> p_behaviour_policy;
    ActionValueTable value_function_;
    RandomPolicy random_policy;
    StateActionMap<double> cumulative_sampling_ratios;
    StateActionMap<double> deltas;
    StateActionMap<long> visit_counts;
    long min_visit = 0;
};

} // namespace rl


