#pragma once

#include "rl/Policy.h"
#include "rl/ActionValueFunction.h"

namespace rl {

/**
 * A policy that is e-greedy with respect to a state-action value function (Q).
 *
 * I'm not sure if this policy has much of a use-case. I originally created it for MCEvaluator3,
 * however, it wasn't really what I was after- what I really needed was BlendedPolicy. The use
 * of QeGreedyPolicy lead to running out of memory. At the beginning of the evaluation, the value
 * function was initialized to 0 and the e-greedy policy was following this somewhat random value
 * function. In some cases this caused very large trials. For example, in a grid world it would
 * nearly always move up. The up action at the top of the grid would result in transitioning to the
 * same state.
 */
class QeGreedyPolicy : public rl::Policy {
public:
    // Our default e is quite exploratory.
    static constexpr double DEFAULT_E = 0.1;
public:
    explicit QeGreedyPolicy(const ActionValueFunction& value_function) :
    value_function(value_function) {}

    const Action& next_action(const Environment& env, const State& from_state) const override {
        const Action* best_action = nullptr;
        double best_return = std::numeric_limits<double>::lowest();
        for(const Action& a : env.actions()) {
            double retrn = value_function.value(from_state, a);
            if(retrn > best_return) {
                best_return = retrn;
                best_action = &a;
            }
        }
        return *CHECK_NOTNULL(best_action);
    }

    ActionDistribution
    possible_actions(const Environment& e, const State& from_state) const override {
        throw std::runtime_error("Not implemented yet.");
        return ActionDistribution();
    }

    void set_e(double e) {
        e_ = e;
    }

    double e() const {
        return e_;
    }

private:
    // Stored as a pointer if assignment operator or move ctr is needed.
    const ActionValueFunction& value_function;
    double e_ = DEFAULT_E;
};

} // namespace rl
