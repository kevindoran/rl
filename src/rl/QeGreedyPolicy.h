#pragma once

#include "rl/Policy.h"
#include "rl/ActionValueTable.h"

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
    explicit QeGreedyPolicy(const ActionValueTable& value_function) :
    value_function(value_function) {}

    explicit QeGreedyPolicy(const ActionValueTable& value_function, double e) :
    value_function(value_function),
    e_(e) {}

    QeGreedyPolicy() = delete;
    QeGreedyPolicy(const QeGreedyPolicy&) = delete;
    QeGreedyPolicy& operator=(const QeGreedyPolicy&) = delete;
    QeGreedyPolicy(QeGreedyPolicy&&) = delete;
    QeGreedyPolicy&& operator=(QeGreedyPolicy&&) = delete;
    ~QeGreedyPolicy() override = default;

    static QeGreedyPolicy create_pure_greedy_policy(const ActionValueTable& value_function) {
        return QeGreedyPolicy(value_function, 0);
    }

    const Action& next_action(const Environment& env, const State& from_state) const override {
        CHECK(!env.is_end_state(from_state));
        double choice = util::random::random_in_range<double>(0, 1);
        bool take_random_action = choice <= e_;
        if(take_random_action) {
            int random_action_index = util::random::random_in_range(0, env.action_count());
            while(!env.is_action_allowed(from_state, env.action(random_action_index))) {
                random_action_index = util::random::random_in_range(0, env.action_count());
            }
            CHECK(env.is_action_allowed(from_state, env.action(random_action_index)));
            return env.action(random_action_index);
        }
        const Action* best_action = nullptr;
        double best_return = std::numeric_limits<double>::lowest();
        for(const Action& a : env.actions()) {
            if(!env.is_action_allowed(from_state, a)) {
                continue;
            }
            double retrn = value_function.value(from_state, a);
            if(retrn > best_return) {
                best_return = retrn;
                best_action = &a;
            }
        }
        return *CHECK_NOTNULL(best_action);
    }

    ActionDistribution
    possible_actions(const Environment& env, const State& from_state) const override {
        if(env.is_end_state(from_state)) {
            return ActionDistribution();
        }
        return ActionDistribution::single_action(next_action(env, from_state));
    }

    void set_e(double e) {
        CHECK_LE(e, 1.0);
        e_ = e;
    }

    double e() const {
        return e_;
    }

private:
    // Stored as a pointer if assignment operator or move ctr is needed.
    const ActionValueTable& value_function;
    double e_ = DEFAULT_E;
};

} // namespace rl
