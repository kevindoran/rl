#ifndef REINFORCEMENT_SUTONBARTOEXERCISES_H
#define REINFORCEMENT_SUTONBARTOEXERCISES_H

#include "rl/MappedEnvironment.h"
#include "rl/GridWorld.h"
#include "grid/Grid.h"

namespace rl {
namespace test {


class Exercise4_1 {
public:
    static const int GRID_WIDTH = 4;
    static const int GRID_HEIGHT = 4;

    static GridWorld<GRID_WIDTH, GRID_HEIGHT>
    create_grid_world() {
        // Setup.
        rl::GridWorld<GRID_HEIGHT, GRID_WIDTH> grid_world;
        const grid::Position top_left{0, 0};
        const grid::Position bottom_right{GRID_HEIGHT-1, GRID_WIDTH-1};
        grid_world.environment().mark_as_end_state(grid_world.pos_to_state(top_left));
        grid_world.environment().mark_as_end_state(grid_world.pos_to_state(bottom_right));
        grid_world.environment().set_all_rewards_to(-1.0);
        grid_world.environment().build_distribution_tree();
        return grid_world;
    }
};

/**
 * Exercise 4.2: Jack's Car Rental.
 *
 *
 *  Location1             Location2
 *   - car_count            - car_count
 */
class Exercise4_2 {
public:

    void create_environment() {

    }

};

} // namespace test
} // namespace rl

#endif //REINFORCEMENT_SUTONBARTOEXERCISES_H
