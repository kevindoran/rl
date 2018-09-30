#ifndef REINFORCEMENT_POLICY_H
#define REINFORCEMENT_POLICY_H

#include "core/Environment.h"
#include "core/ValueFunction.h"

namespace rl {






class Policy {
public:
    struct ActionDistribution {
        using WeightMap = std::unordered_map<const Action*, int>;

        static ActionDistribution single_action(const Action& a) {
            static const int weight = 1;
            WeightMap m{ {&a, weight} };
            return {m, weight};
        }

        WeightMap weight_map;
        int total_weight;
    };

public:
    virtual const Action& next_action(const Environment& e, const State& from_state) const = 0;

    virtual ActionDistribution possible_actions(const Environment& e,
                                                const State& from_state) const = 0;


    // Should we make this pure? If it is not pure, Policy might not be considered a virtual class.
    // Is there issues with that?
    virtual ~Policy() = default;
};

class DeterministicLambdaPolicy : public Policy {
public:
    using Callback = std::function<const Action&(const Environment&, const State&)>;

    explicit DeterministicLambdaPolicy(Callback fctn) : fctn_(std::move(fctn))
    {}

    const Action& next_action(const Environment& e, const State& from_state) const override {
        return fctn_(e, from_state);
    }

    ActionDistribution possible_actions(const Environment& e,
                                        const State& from_state) const override {
        const Action& a = next_action(e, from_state);
        return ActionDistribution::single_action(next_action(e, from_state));
    }

private:
    Callback fctn_;
};

class PolicyEvaluation {
public:
    virtual ValueFunction evaluate(Environment& e, const Policy& p) = 0;
    virtual ~PolicyEvaluation() = default;
};

} // namespace rl

#endif //REINFORCEMENT_POLICY_H
