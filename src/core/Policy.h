#ifndef REINFORCEMENT_POLICY_H
#define REINFORCEMENT_POLICY_H

#include <memory>

#include "core/Environment.h"
#include "core/ValueFunction.h"

namespace rl {


class Policy {
public:
    class ActionDistribution {
    public:
        using WeightMap = std::unordered_map<const Action*, int>;
        using Actions = std::vector<std::reference_wrapper<const Action>>;

        static ActionDistribution single_action(const Action& a) {
            static const int weight = 1;
            WeightMap m{ {&a, weight} };
            return {m, weight};
        }

        void add_action(const Action& a, int weight=1) {
            bool already_exists = static_cast<bool>(weight_map.count(&a));
            Expects(!already_exists);
            weight_map[&a] = weight;
            total_weight += weight;
        }

        WeightMap weight_map{};
        int total_weight = 0;
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
    virtual ValueFunction evaluate(Environment& e, const Policy& p) = 0;
    virtual ~PolicyEvaluation() = default;
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
    virtual std::unique_ptr<Policy> improve(const Policy& policy) const = 0;
    virtual ~PolicyImprovement() = default;
};

} // namespace rl

#endif //REINFORCEMENT_POLICY_H
