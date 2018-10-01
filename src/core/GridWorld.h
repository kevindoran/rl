#ifndef REINFORCEMENT_GRIDWORLD_H
#define REINFORCEMENT_GRIDWORLD_H

#include "core/Environment.h"
#include "core/Grid.h"

namespace rl {

/**
 * An environment representing an WxH grid where each grid entry represents a state.
 *
 * 2D grid where states are grid positions, actions are up, down, left and right and rewards are
 * tied to states.
 *
 *   +---+---+---+---+
 *   | 0 | 1 | 2 | 3 |
 *   +---+---+---+---+
 *   | 4 | 5 | 6 | 7 |
 *   +---+---+---+---+
 *
 * Actions
 * From each state up to 5 actions are available representing up, down, left, right, and movements.
 * If a grid entry doesn't have a neighbour on one side, then the corresponding action to move in
 * that direction will not be available.
 *
 * Next states
 * The next state after an action is guaranteed to move in the intended direction. Thus, each
 * (state, action) tuple has a 1-1 correspondence with a next action.
 *
 * Rewards
 * Rewards are assigned to states. The reward given to a transition is the reward assigned to the
 * state.
 *
 * End state
 * One or more grid positions can be marked as end states.
 *
 */
template<int HEIGHT, int WIDTH>
class GridWorld {

public:
    using GridType = grid::Grid<HEIGHT, WIDTH>;

    GridWorld() {
        // Add states and rewards (rewards are 1-1 with states).
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                grid::Position p{y, x};
                // Add state.
                std::string state_name = p.to_string();
                State &s = environment_.add_state(state_name);
                Ensures(s == pos_to_state(p));
                // Add reward.
                Reward &r = environment_.add_reward(DEFAULT_REWARD, state_name);
                Ensures(r.id() == reward_at(p).id());
            }
        }

        // Add actions. There are just 4 actions.
        for (grid::Direction d : grid::directions) {
            Action &a = environment_.add_action(grid::to_string(d));
            Ensures(a.id() == dir_to_action_id(d));
        }

        // Add transitions.
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                grid::Position from_pos{y, x};
                // Tile might not always be the same as the state ID.
                const State& from_state = pos_to_state(from_pos);
                for (grid::Direction d : grid::directions) {
                    grid::Position to_pos = from_pos.adj(d);
                    if (!grid_.is_valid(to_pos)) {
                        continue;
                    }
                    const State& to_state = pos_to_state(to_pos);
                    const Action& action = dir_to_action(d);
                    const Reward& reward = reward_at(to_pos);
                    // Do we need a mapping for these?
                    Transition t{from_state, to_state, action, reward};
                    environment_.add_transition(t);
                }
            }
        }
    }

    const State& pos_to_state(grid::Position p) const {
        // Making some assumptions on the ids and enum values matching. Could use a map instead.
        return environment_.state(grid_.to_id(p));
    }

    grid::Position state_to_pos(const State& state) const {
        return grid_.to_position(state.id());
    }

    const Action& dir_to_action(grid::Direction d) const {
        return environment_.action(dir_to_action_id(d));
    }

    ID dir_to_action_id(grid::Direction d) const {
        // Making some assumptions on the ids and enum values matching. Could use a map instead.
        return d;
    }

    /**
     * Returns the reward that is given when moving to \c target_state.
     *
     * This method highlights the restriction of GridWorld- rewards are determined only by the
     * target state and do not have any probability distribution.
     */
    const Reward& reward_at(grid::Position target_state) const {
        // Making some assumptions on the ids and enum values matching. Could use a map instead.
        return environment_.reward(pos_to_state(target_state).id());
    }

    Reward& reward_at(grid::Position target_state) {
        return const_cast<Reward&>(static_cast<const GridWorld*>(this)->reward_at(target_state));
    }

    Environment& environment() {
        return environment_;
    }

    const GridType& grid() const {
        return grid_;
    }

    GridType& grid() {
        return const_cast<GridType&>(static_cast<const GridWorld*>(this)->grid());
    }

    /// Some conveniences:
    grid::Position current_pos() {
        return state_to_pos(environment_.current_state());
    }

public:
    const int DEFAULT_REWARD = 0;

private:
    Environment environment_;
    GridType grid_;
};

} // namespace rl

#endif //REINFORCEMENT_GRIDWORLD_H
