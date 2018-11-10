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

bool greater_than(double val1, double val2, double by_at_least);

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
            // This weight_map_ was first needed by importance sampling MC evaluation- it needs
            // a mapping from action->weight. A benefit is the simplification of the weight_map()
            // function.
            auto res = weight_map_.emplace(&a, weight);
            bool inserted_without_override = res.second;
            CHECK(inserted_without_override);
        }

        const Action& random_action() const {
            const Action& result = *CHECK_NOTNULL(action_list_.random());
            return result;
        }

        const Action& any() const {
            CHECK(!action_list_.entries().empty());
            const Action& a = *CHECK_NOTNULL(action_list_.entries()[0].data());
            return a;
        }

        Weight total_weight() const {
            return action_list_.total_weight();
        }

        /**
         * Returns the probability weight of the given action being chosen.
         *
         * If the given action has no chance of being chosen, 0 will be returned.
         */
        Weight weight(const Action& action) const {
            auto it = weight_map_.find(&action);
            if(it == std::end(weight_map_)) {
                return 0;
            } else {
                double res = it->second;
                return res;
            }
        }

        ID action_count() const {
            return static_cast<ID>(action_list_.entries().size());
        }

        bool empty() const {
            return action_count() == 0;
        }

        const WeightMap& weight_map() const {
            return weight_map_;
        }

    private:
        using ActionList = DistributionList<const Action, Weight>;
        ActionList action_list_{};
        WeightMap weight_map_{};
    };

public:
    virtual const Action& next_action(const Environment& e, const State& from_state) const = 0;

    // TODO: decide behaviour for what should happen when there are no actions.
    // note: the return type could be changed to shared_ptr<ActionDistribution> to allow for the
    // option of returning an existing object rather than copying.
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
     * \returns the number of steps carried out so far.
     */
    virtual long steps_done() const = 0;

    /**
     * This method was created to allow evaluators some flexibility over how they consider
     * a value function to have converged. A simple implementation would be to check the most
     * recent delta against the delta threshold.
     *
     * \returns \c true if the value function meets the criteria such that it can be considered
     *          converged; \c false otherwise.
     */
    virtual bool finished() const = 0;

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
class PolicyImprover {
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

    virtual PolicyEvaluator& policy_evaluator() = 0;
    virtual const PolicyEvaluator& policy_evaluator() const = 0;

    virtual ~PolicyImprover() = default;
};

} // namespace rl
