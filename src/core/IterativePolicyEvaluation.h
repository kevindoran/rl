#ifndef REINFORCEMENT_ITERATIVEPOLICYEVALUATION_H
#define REINFORCEMENT_ITERATIVEPOLICYEVALUATION_H

#include <limits>

#include "core/Policy.h"

namespace rl {

class IterativePolicyEvaluation : public PolicyEvaluation {
public:

    ValueFunction evaluate(Environment& e, const Policy& p) override {
        // error = inf
        // while(error > threshold)
        //    for(s in states)
        //        old_s = values[s]
        //        value[s] = 0
        //        for( t in transitions)
        //            value[s] += (t.reward + value[s']) * t.prob)
        //        error = max(error, (old_s - value[s])
        double error = std::numeric_limits<double>::max();
        ValueFunction res(e.state_count());
        while(error < delta_threshold_) {
            error = 0;
        }
        return res;
    }

    void set_delta_threshold(double delta_threshold) {
        delta_threshold_ = delta_threshold;
    }



private:
    double delta_threshold_;
};

} // namespace rl

#endif //REINFORCEMENT_ITERATIVEPOLICYEVALUATION_H
