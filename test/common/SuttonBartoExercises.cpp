#include "SuttonBartoExercises.h"

namespace {
    int R = grid::Direction::RIGHT;
    int D = grid::Direction::DOWN;
    int L = grid::Direction::LEFT;
    int U = grid::Direction::UP;
} // namespace

namespace rl {
namespace test {

const std::unordered_set<int> Exercise4_1::optimal_actions_[Exercise4_1::GRID_WIDTH * Exercise4_1::GRID_HEIGHT] = {
    {   }, {  L    }, {  L    }, {D,L},
    {  U}, {  L,  U}, {D,L,R,U}, {D  },
    {  U}, {D,L,R,U}, {D,  R  }, {D  },
    {R,U}, {    R  }, {    R  }, {   }
};

} // namespace test
} // namespace rl

