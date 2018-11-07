#pragma once

#include "rl/Policy.h"

#include <glog/logging.h>

namespace rl {
namespace impl {
class PolicyEvaluator;
} // namespace impl

/**
 * A convenient base class for PolicyEvaluation implementations.
 *
 * Handles the trivial handling of setting and getting delta_threshold
 * and discount_rate.
 */
class impl::PolicyEvaluator : public virtual rl::PolicyEvaluator {

public:
    static constexpr double DEFAULT_DELTA_THRESHOLD = 0.00001;
    static constexpr double DEFAULT_DISCOUNT_RATE = 1.0;

    void initialize(const rl::Environment& e, const rl::Policy& p) override {
        env_ = &e;
        policy_ = &p;
        steps_ = 0;
        most_recent_delta_ = std::numeric_limits<double>::max();
    }

    void run() override {
        LOG_IF(ERROR, finished()) << "The evaluation end criteria is met before starting.";
        while(most_recent_delta_ > delta_threshold_) {
            long previous_step_count = steps_done();
            step();
            CHECK_EQ(steps_done(), previous_step_count + 1);
        }
    }

    long steps_done() const override {
        return steps_;
    }

    bool finished() const override {
        return most_recent_delta_ < delta_threshold_;
    }

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
    const rl::Environment* env_ = nullptr;
    const rl::Policy* policy_ = nullptr;
    long steps_ = 0;
    double most_recent_delta_ = std::numeric_limits<double>::max();
    double delta_threshold_ = DEFAULT_DELTA_THRESHOLD;
    double discount_rate_ = DEFAULT_DISCOUNT_RATE;
};

} // namespace rl
