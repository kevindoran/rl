#pragma once

#include <memory>
#include <glog/logging.h>

#include "rl/Environment.h"
#include "rl/ValueFunction.h"
#include "rl/DistributionList.h"

namespace rl {


class Policy {
public:
    class ActionDistribution {
    public:
        using WeightMap = std::unordered_map<const Action*, long>;

        static ActionDistribution single_action(const Action& a) {
            ActionDistribution dist;
            long weight = 1;
            dist.add_action(a, weight);
            return dist;

        }

        void add_action(const Action& a, long weight=1) {
            action_list_.add(weight, &a);
        }

        const Action& random_action() const {
            const Action& result = *CHECK_NOTNULL(action_list_.random());
            return result;
        }

        long total_weight() const {
            return action_list_.total_weight();
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
        using ActionList = DistributionList<const Action>;
        ActionList action_list_{};
    };

public:
    virtual const Action& next_action(const Environment& e, const State& from_state) const = 0;

    virtual ActionDistribution possible_actions(const Environment& e,
                                                const State& from_state) const = 0;

    // Should we make this pure? If it is not pure, Policy might not be considered a virtual class.
    // Is there issues with that?
    virtual ~Policy() = default;
};

// TODO: edit the methods evaluate() and improve() to accept some options that allow stopping
// conditions to be controlled. It would also be nice to be able to interact with the in-progress
// results after a certain number of loops or units of time.

class PolicyEvaluation {
public:
    virtual ValueFunction evaluate(const Environment& e, const Policy& p) = 0;
    virtual ~PolicyEvaluation() = default;

    virtual void set_discount_rate(double discount_rate) = 0;
    virtual double discount_rate() const = 0;

};

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
