#pragma once

#include "rl/Policy.h"

namespace rl {
namespace impl {
class PolicyEvaluation;
} // namespace impl

/**
 * A convenient base class for PolicyEvaluation implementations.
 *
 * Handles the trivial handling of setting and getting delta_threshold
 * and discount_rate.
 */
class impl::PolicyEvaluation : public rl::PolicyEvaluation {

public:
    static constexpr double DEFAULT_DELTA_THRESHOLD = 0.00001;
    static constexpr double DEFAULT_DISCOUNT_RATE = 1.0;

    void set_delta_threshold(double delta_threshold) override {
        delta_threshold_ = delta_threshold;
    }

    double delta_threshold() const override {
        return delta_threshold_;
    }

    void set_discount_rate(double discount_rate) override {
        discount_rate_ = discount_rate;
    }

    double discount_rate() const override {
        return discount_rate_;
    }

protected:
    double delta_threshold_ = DEFAULT_DELTA_THRESHOLD;
    double discount_rate_ = DEFAULT_DISCOUNT_RATE;
};

} // namespace rl
