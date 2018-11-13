#pragma once

#include "impl/PolicyEvaluator.h"
#include "Trial.h"
#include "StateActionMap.h"
#include "RandomPolicy.h"

namespace rl {

/**
 * On-policy temporal difference evaluator.
 *
 * Properties:
 *   * every-visit
 *   * exploring starts (to allow visiting all states)
 *   * Expected-Sarsa style TD error
 */
class TDEvaluator : public ActionBasedEvaluator, public impl::PolicyEvaluator {
public:
    static const int MIN_VISIT = 100;

public:
    void initialize(const Environment& env, const Policy& policy) override;
    void step() override;
    const ActionValueFunction& value_function() const override;
    bool finished() const override;

private:
    void update_value_fctn(const Trace& trace);

    /**
     * Calculate the expected return from the given state when following the policy being evaluated.
     *
     * This method flattens the state-action value function into a state value for a given state.
     *
     * The calculation is a simple sum, over a: prob(a | state) * action_value(a)
     */
    double state_value(const State& state) const;

private:
    ActionValueFunction value_function_;
    StateActionMap<double> deltas;
    StateActionMap<long> visit_counts;
    long min_visit = 0;
};

} // namespace rl

