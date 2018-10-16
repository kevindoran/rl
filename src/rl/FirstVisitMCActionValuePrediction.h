#pragma once

#include "rl/ActionValueFunction.h"
#include "rl/Policy.h"

namespace rl {

// Long enough name?
class FirstVisitMCActionValuePrediction : ActionValuePolicyEvaluation {

public:
    static constexpr double DEFAULT_DELTA_THRESHOLD = 0.00001;
    static constexpr double DEFAULT_DISCOUNT_RATE = 1.0;

    ActionValueFunction evaluate(const Environment& e, const Policy& p) override {
        // We will use first-visit & exploring starts.

        return ActionValueFunction(0,0);
    }

    void set_discount_rate(double discount_rate) override {
        throw std::runtime_error("This evaluator only supports episodic tasts.");
    }

    double discount_rate() const override {
        return 1.0;
    }

    void set_delta_threshold(double delta_threshold) override {
        delta_threshold_ = delta_threshold;
    }

    double delta_threshold() const override {
        return delta_threshold_;
    }

protected:
    double delta_threshold_ = DEFAULT_DELTA_THRESHOLD;
    double discount_rate_ = DEFAULT_DISCOUNT_RATE;

};

} // namespace rl
