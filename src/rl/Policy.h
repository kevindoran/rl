#pragma once

#include <memory>
#include <glog/logging.h>

#include "rl/Environment.h"
#include "rl/ValueFunction.h"
#include "rl/DistributionList.h"
#include "rl/ActionValueFunction.h"

namespace rl {

double error_as_factor(double prev, double updated);

int compare(double val1, double val2, double error_factor);

/**
 * For every state in an Environment, Policy defines actions to be taken (with probability).
 */
class Policy {
public:
    class ActionDistribution {
    public:
        using WeightMap = std::unordered_map<const Action*, Weight>;

        static ActionDistribution single_action(const Action& a) {
            ActionDistribution dist;
            Weight weight = 1;
            dist.add_action(a, weight);
            return dist;

        }

        void add_action(const Action& a, Weight weight=1) {
            action_list_.add(weight, &a);
        }

        const Action& random_action() const {
            const Action& result = *CHECK_NOTNULL(action_list_.random());
            return result;
        }

        Weight total_weight() const {
            return action_list_.total_weight();
        }

        ID action_count() const {
            return static_cast<ID>(action_list_.entries().size());
        }

        bool empty() const {
            return action_count() == 0;
        }

        WeightMap weight_map() const {
            WeightMap ans;
            std::transform(
                    std::begin(action_list_.entries()), std::end(action_list_.entries()),
                    std::inserter(ans, ans.end()),
                    [](const auto& entry) {
                        return WeightMap::value_type{entry.data(), entry.weight()};
                    }
                );
            return ans;
        }

    private:
        using ActionList = DistributionList<const Action, Weight>;
        ActionList action_list_{};
    };

public:
    virtual const Action& next_action(const Environment& e, const State& from_state) const = 0;

    // TODO: decide behaviour for what should happen when there are no actions.
    virtual ActionDistribution possible_actions(const Environment& e,
                                                const State& from_state) const = 0;

    virtual ~Policy() = default;
};


/**
 * Calculates the (state) value function for a Policy.
 */
class PolicyEvaluator {
public:

    //----------------------------------------------------------------------------------------------
    // Setup & run
    //----------------------------------------------------------------------------------------------

    /**
     * Initializes the evaluator to evaluate policy \c p in environment \e. Resets all results.
     */
    virtual void initialize(const Environment& e, const Policy& p) = 0;

    /**
     * Carry out a single iteration of the evaluation algorithm.
     */
    virtual void step() = 0;

    /**
     * Run the evaluation algorithm until an end condition is reached.
     */
    virtual void run() = 0;

    //----------------------------------------------------------------------------------------------
    // Results
    //----------------------------------------------------------------------------------------------

    /**
     * \returns a measure for how much the value function changed in the most recent iteration.
     */
    virtual double delta() const = 0;

    /**
     * \returns the number of steps carried out so far.
     */
    virtual long steps_done() const = 0;

    //----------------------------------------------------------------------------------------------
    // Settings
    //----------------------------------------------------------------------------------------------
    virtual void set_discount_rate(double discount_rate) = 0;
    virtual double discount_rate() const = 0;
    virtual void set_delta_threshold(double max_delta) = 0;
    virtual double delta_threshold() const = 0;

    virtual ~PolicyEvaluator() = default;
};


class StateBasedEvaluator : public virtual PolicyEvaluator {
public:
    /**
     * \returns the current estimate of the policy's value function.
     */
    virtual const ValueFunction& value_function() const = 0;

    ~StateBasedEvaluator() override = default;
};

/**
 * Calculates the (action) value function for a Policy.
 */
class ActionBasedEvaluator : public virtual PolicyEvaluator {
public:

    virtual const ActionValueFunction& value_function() const = 0;

    ~ActionBasedEvaluator() override = default;
};

// note: Having this template method is far more convenient that having all sub-types define a:
//      ValueType init_and_run(Environment&,Policy&)
// method. The presence of the specific ValueType such as ValueFunction prevent the method from
// being declared in the PolicyEvaluator interface.
template<typename EvaluatorT>
// note: return value or const ref here?
const auto& evaluate(EvaluatorT& evaluator, const Environment& env, const Policy& policy) {
    evaluator.initialize(env, policy);
    evaluator.run();
    return evaluator.value_function();
}


/**
 * Calculates the optimal policy (or approximation to it) for an environment.
 */
class PolicyImprovement {
public:
    // A conclusion based on the logic from:
    // https://en.wikipedia.org/wiki/One-shot_deviation_principle, and
    // https://mathoverflow.net/a/44685
    // is that if there is an optimal policy, then there is a deterministic optimal policy.
    // Therefore, we could choose to update this return type to be a DeterministicPolicy rather than
    // the abstract super type, Policy. I'll leave it as it unless it becomes troublesome. A more
    // flexible return type might allow for a more efficient policy representation than what is
    // possible with the map used by DeterministicPolicy.
    virtual std::unique_ptr<Policy> improve(const Environment& env, const Policy& policy) const = 0;

    virtual ~PolicyImprovement() = default;
};

} // namespace rl
