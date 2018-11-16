#include "Example6_6.h"

namespace {
int R = grid::Direction::RIGHT;
int D = grid::Direction::DOWN;
int L = grid::Direction::LEFT;
int U = grid::Direction::UP;
} // namespace

namespace rl {
namespace test {
namespace suttonbarto {

const std::vector<grid::Position> Example6_6::SAFE_ROUTE {
        {3, 0}, {2, 0}, {1, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7},
        {0, 8}, {0, 9}, {0, 10}, {0, 11}, {1, 11}, {2, 11}, {3, 11}
};

const std::vector<grid::Position> Example6_6::OPTIMAL_ROUTE {
        {3, 0}, {2, 0}, {1, 0}, {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7},
        {0, 8}, {0, 9}, {0, 10}, {0, 11}, {1, 11}, {2, 11}, {3, 11}
};


const std::unordered_set<int> Example6_6::optimal_actions_[Example6_6::HEIGHT *
                                                           Example6_6::WIDTH] = {
{R, D}, {R, D}, {R, D}, {R, D}, {R, D}, {R, D}, {R, D}, {R, D}, {R, D}, {R, D}, {R, D}, {  D},
{R, D}, {R, D}, {R, D}, {R, D}, {R, D}, {R, D}, {R, D}, {R, D}, {R, D}, {R, D}, {R, D}, {  D},
{R,  }, {R   }, {R   }, {R   }, {R   }, {R   }, {R   }, {R   }, {R   }, {R   }, {R   }, {  D},
// The tiles representing the cliff can't actually be reached. Moving toward the cliff just diverts
// to (3,0). Thus, we can allow any action in these states for the purpose of testing optimal
// policies.
{U   }, {R,U,L,D}, {R,U,L,D}, {R,U,L,D}, {R,U,L,D}, {R,U,L,D}, {R,U,L,D}, {R,U,L,D}, {R,U,L,D},
                                                                  {R,U,L,D}, {R,U,L,D}, {   } };

} // namespace suttonbarto
} // namespace test
} // namespace rl