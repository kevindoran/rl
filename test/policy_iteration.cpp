#include "gtest/gtest.h"

#include <unordered_set>

#include "rl/GridWorld.h"
#include "grid/Grid.h"
#include "rl/Policy.h"
#include "common/SuttonBartoExercises.h"
#include "rl/PolicyIteration.h"
#include "rl/RandomGridPolicy.h"

/**
 * This test calculates the optimum policy for the gridworld described in exercise 4.1 in
 * (Sutton & Barto, 2018). The initial policy is the random grid policy.
 *
 * The optimum policy is:
 * (actions available: U = up, D = down, L = left, R = right)
 *
 * X  L  L  DL
 * U  LU DL D
 * U  RU RD D
 * RU R  R  X
 *
 *
 */
TEST(PolicyIteration, sutton_barto_exercise_4_1) {
    auto grid_world = rl::test::Exercise4_1::create_grid_world();
    const rl::MappedEnvironment& env = grid_world.environment();
    rl::RandomGridPolicy random_policy(grid_world);
    rl::PolicyIteration policy_improver;
    std::unique_ptr<rl::Policy> p_policy = policy_improver.improve(env, random_policy);
    ASSERT_TRUE(p_policy);

    int R = grid::Direction::RIGHT;
    int D = grid::Direction::DOWN;
    int L = grid::Direction::LEFT;
    int U = grid::Direction::UP;
    std::unordered_set<int> optimal_actions[] = {
            {   }, {  L}, {  L}, {D,L},
            {  U}, {L,U}, {D,L}, {D,L},
            {  U}, {R,U}, {R,D}, {  D},
            {R,U}, {  R}, {  R}, {   }
    };

    for(int i = 0; i < grid_world.environment().state_count(); i++) {
        rl::Policy::ActionDistribution action_dist = p_policy->possible_actions(env, env.state(i));
        auto weight_map = action_dist.weight_map();
        // End states shouldn't have an action.
        if(optimal_actions[i].empty()) {
            ASSERT_TRUE(weight_map.empty());
            // There is nothing left to test for end states.
            continue;
        }
        // The policy must not contain more actions that the optimal policy set.
        ASSERT_GE(optimal_actions[i].size(), weight_map.size());
        // The policy's actions should be an optmal policy.
        for(auto entry : weight_map) {
            const rl::Action& policy_action = *CHECK_NOTNULL(entry.first);
            ASSERT_TRUE(optimal_actions[i].count(grid_world.action_to_dir(policy_action)));
        }
    }
}

TEST(PolicyIteration, sutton_barto_exercise_4_2) {

}