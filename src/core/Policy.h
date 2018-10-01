#ifndef REINFORCEMENT_POLICY_H
#define REINFORCEMENT_POLICY_H

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


class PolicyEvaluation {
public:
    virtual ValueFunction evaluate(Environment& e, const Policy& p) = 0;
    virtual ~PolicyEvaluation() = default;
};

} // namespace rl

#endif //REINFORCEMENT_POLICY_H
