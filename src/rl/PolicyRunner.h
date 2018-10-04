#pragma once

#include "rl/Policy.h"
#include "rl/MappedEnvironment.h"

namespace rl {

class PolicyRunner {
public:
    struct Results {
        int action_count = 0;
        double accumulated_reward = 0;
        bool action_limit_reached = false;
        // Other things such as state history trial.
    };

    static const int NO_ACTION_LIMIT = -1;
    static Results run(MappedEnvironment& e, Policy& p, int action_limit=NO_ACTION_LIMIT) {
        e.restart();
        Results res{};
        while(true) {
            if(action_limit != NO_ACTION_LIMIT and res.action_count > action_limit) {
                res.action_limit_reached = true;
                break;
            }
            const Action& next_action = p.next_action(e);
            e.execute_action(next_action);
            res.action_count++;
        }
        res.accumulated_reward = e.accumulated_reward();
        return res;
    }
};

} // namespace rl
