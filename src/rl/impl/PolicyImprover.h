#pragma once

#include <rl/Policy.h>

namespace rl {
namespace impl {
class PolicyImprover;
} // namespace impl

class impl::PolicyImprover : public virtual rl::PolicyImprover {
public:
    static constexpr double DEFAULT_DELTA_THRESHOLD = 0.00001;
    static constexpr double DEFAULT_DISCOUNT_RATE = 1.0;

public:
    void set_discount_rate(double discount_rate) override {
        discount_rate_ = discount_rate;
    }

    double discount_rate() const override {
        return discount_rate_;
    }

    void set_delta_threshold(double max_delta) override {
        delta_threshold_ = max_delta;
    }

    double delta_threshold() const override {
        return delta_threshold_;
    }

protected:
    double delta_threshold_ = DEFAULT_DELTA_THRESHOLD;
    double discount_rate_ = DEFAULT_DISCOUNT_RATE;
};

} // namespace rl