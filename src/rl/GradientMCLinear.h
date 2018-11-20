#pragma once

#include <Eigen/Dense>
#include "impl/PolicyEvaluator.h"
#include "Trial.h"

namespace rl {

class ValueFunction {
public:
    /**
     * For the moment, it seems that the input state cannot be an end state, else the ValueFunction
     * will need to hold on to an environment reference and check each time against the end state
     * list.
     */
    virtual double value(const State& s, const Eigen::VectorXd& weights) const = 0;
    virtual double value(const State& s) const = 0;
    virtual Eigen::VectorXd derivative(const State& s, const Eigen::VectorXd& weights) const = 0;
    virtual Eigen::VectorXd derivative(const State& s) const = 0;
    virtual void set_weights(const Eigen::VectorXd& weights) = 0;
    virtual const Eigen::VectorXd& weights() const = 0;
    virtual Eigen::VectorXd& weights() = 0;
    virtual ~ValueFunction() = default;
};

class StateAggregateValueFunction : public ValueFunction {
public:
    StateAggregateValueFunction(int group_count, std::vector<int> state_to_group_map) :
        group_count_(group_count),
        state_to_group_map_(std::move(state_to_group_map)),
        weights_(Eigen::VectorXd::Zero(group_count_)){
        for(int group_id : state_to_group_map) {
            CHECK_LE(group_id, group_count);
        }
    }

    double value(const State& s, const Eigen::VectorXd& weights) const override {
        CHECK_EQ(weights.rows(), group_count_);
        return weights(state_to_group_map_.at(s.id()));
    }

    double value(const State& s) const override {
        return value(s, weights_);
    }

    Eigen::VectorXd derivative(const State& s, const Eigen::VectorXd& weights) const override {
        CHECK_EQ(weights.rows(), group_count_);
        Eigen::VectorXd ans = Eigen::VectorXd::Zero(group_count_);
        ans(state_to_group_map_.at(s.id())) = 1.0;
        return ans;
    }

    Eigen::VectorXd derivative(const State& s) const override {
        return derivative(s, weights_);
    }

    const Eigen::VectorXd& weights() const override {
        return weights_;
    }

    Eigen::VectorXd& weights() override {
        return const_cast<Eigen::VectorXd&>(
                static_cast<const StateAggregateValueFunction*>(this)->weights());
    }

    /**
     *
     * \param weights Normally, weights would be taken by value and move-assigned inside the
     * function, however, Eigen::Matrix doesn't support move or move assignment.
     */
    void set_weights(const Eigen::VectorXd& weights) override {
        weights_ = weights;
    }

private:
    int group_count_;
    std::vector<int> state_to_group_map_;
    Eigen::VectorXd weights_;
};

// doesn't extend PolicyEvaluator, as it's not clear if the abstractions are shared.
// For the moment, using a very simple api. The whole set of evaluators need to be revisited
// to see what common abstractions there might be.
class GradientMCLinear {
public:
    // Very simple stopping criteria to begin with.
    static constexpr int DEFAULT_ITERATION_COUNT = 100000;
    static constexpr double DEFAULT_STEP_SIZE = 2e-5;

public:
    /**
     * Moves the weights of the given value function towards values which cause thevalue function to
     * better represent the state value function of the given policy operating in the given
     * environment.
     *
     * 'Better' is defined as minimizing the Value Error:
     *    VE = sum(for s in the state set)[on_policy_distribution(s) * (v_pi(s) - v(s, w))^2]
     *
     *  Where v_pi is the true value function, and v is the approximate value function which is
     *  given.
     *
     *  As we do not know v_pi, it will be estimated from running trials. This estimation will also
     *  subsume the on-policy distribution.
     *
     * \param env [in] The environment where the policy is to operate in.
     * \param policy [in] The policy to evaluate.
     * \param value_function [in, out] The value function in functional form. The process of
     *      evaluation involves setting the value function's weights to values which make the
     *      value function accurate.
     */
    void evaluate(const Environment& env, const Policy& policy, ValueFunction& value_function) {
        for(int i = 0; i < iterations_; i++) {
            Trace trace = run_trial(env, policy);
            double reward = trace.back().reward;
            for(auto it = std::next(std::crbegin(trace)); it != std::crend(trace); ++it) {
                const TimeStep& ts = *it;
                const double current_val = value_function.value(ts.state);
                // no discounting:
                const double trial_val = reward;
                const double error = trial_val - current_val;
                DLOG(INFO) << "Weights before update:" << std::endl << value_function.weights();
                value_function.weights() += step_size_ * error * value_function.derivative(ts.state);
                DLOG(INFO) << "Weights after update:" << std::endl << value_function.weights();
            }
        }
    }

private:
    int iterations_ = DEFAULT_ITERATION_COUNT;
    double step_size_ = DEFAULT_STEP_SIZE;
};

} // namespace rl

