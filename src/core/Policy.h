#ifndef REINFORCEMENT_POLICY_H
#define REINFORCEMENT_POLICY_H

#include "core/Environment.h"
#include "core/ValueFunction.h"

namespace rl {




class Policy {
public:
    // Return an action ID or Action?
    virtual const Action& next_action(const Environment& e) = 0;
    // Should we make this pure? If it is not pure, Policy might not be considered a virtual class.
    // Is there issues with that?
    virtual ~Policy() = default;
};

class LambdaPolicy : public Policy {
public:
    using Callback = std::function<const Action&(const Environment&)>;

    explicit LambdaPolicy(Callback fctn) : fctn_(fctn)
    {}
    const Action& next_action(const Environment& e) override {
        return fctn_(e);
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
