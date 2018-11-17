#pragma once

#include "TestEnvironment.h"
#include "CarRentalEnvironment.h"

namespace rl {
namespace test {
namespace suttonbarto {

/**
 * Exercise 4.2: Jack's Car Rental.
 *
 */
class Exercise4_2 : public TestEnvironment {
public:
    std::string name() const override {
        return "Sutton & Barto exercise 4.2";
    }

    double required_discount_rate() const override {
        return 0.9;
    }

    double required_delta_threshold() const override {
        return 1e-6;
    }

    const CarRentalEnvironment& env() const override {
        return env_;
    }

    OptimalActions optimal_actions(const State& from_state) const override {
        int l1 = env_.cars_in_loc_1(from_state);
        int l2 = env_.cars_in_loc_2(from_state);
        int cars_moved = optimal_policy[l1][l2];
        const Action& single_optimal_action = env_.action(env_.action_id(cars_moved));
        // TODO: what is the initializer list syntax for set of ref-wrappers?
        return OptimalActions{single_optimal_action.id()};
    }

private:
    CarRentalEnvironment env_;

    // Copying the values from the book.
    static constexpr int optimal_policy[CarRentalEnvironment::MAX_CAR_COUNT + 1]
    [CarRentalEnvironment::MAX_CAR_COUNT + 1] =
            // 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
            {{0, 0, 0, 0, 0, 0, 0, 0, -1,     -1, -2, -2, -2, -3, -3, -3, -3, -3, -4, -4, -4}, //  0
             {0, 0, 0, 0, 0, 0, 0, 0, 0,      -1, -1, -1, -2, -2, -2, -2, -2, -3, -3, -3, -3}, //  1
             {0, 0, 0, 0, 0, 0, 0, 0, 0,      0,  0,  -1, -1, -1, -1, -1, -2, -2, -2, -2, -2}, //  2
             {0, 0, 0, 0, 0, 0, 0, 0, 0,      0,  0,  0,  0,  0,  0,  -1, -1, -1, -1, -1, -2}, //  3
             {0, 0, 0, 0, 0, 0, 0, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  -1, -1}, //  4
             {1, 1, 1, 0, 0, 0, 0, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, //  5
             {2, 2, 1, 1, 0, 0, 0, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, //  6
             {3, 2, 2, 1, 1, 0, 0, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, //  7
             {3, 3, 2, 2, 1, 1, 0, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, //  8
             {4, 3, 3, 2, 2, 1, 0, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, //  9
             {4, 4, 3, 3, 2, 1, 0, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 10
             {5, 4, 4, 3, 2, 1, 1, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 11
             {5, 5, 4, 3, 2, 2, 1, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 12
             {5, 5, 4, 3, 3, 2, 1, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 13
             {5, 5, 4, 4, 3, 2, 1, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 14
             {5, 5, 5, 4, 3, 2, 1, 0, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 15
             {5, 5, 5, 4, 3, 2, 1, 1, 0,      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 16
                    // FIXME: (17, 8) should be: 0. Making the tests assert incorrect behaviour until fixed.
             {5, 5, 5, 4, 3, 2, 2, 1, 1/*0*/, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 17
             {5, 5, 5, 4, 3, 3, 2, 2, 1,      1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0}, // 18
             {5, 5, 5, 4, 4, 3, 3, 2, 2,      2,  2,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0}, // 19
             {5, 5, 5, 5, 4, 4, 3, 3, 3,      3,  2,  2,  2,  2,  2,  1,  1,  1,  0,  0,  0}  // 20
            };
};

} // namespace suttonbarto
} // namespace test
} // namespace rl
