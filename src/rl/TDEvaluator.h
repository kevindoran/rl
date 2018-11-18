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
    const ActionValueTable& value_function() const override;
    bool finished() const override;

private:
    void update_value_fctn(const Trace& trace);

private:
    ActionValueTable value_function_;
    StateActionMap<double> deltas;
    StateActionMap<long> visit_counts;
    long min_visit = 0;
};

} // namespace rl

