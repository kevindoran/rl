#include "Grid.h"
#include "util/random.h"

namespace grid {

// Example:
// change in x coordinate when moving in Direction d = dx[d].
int Position::dx[] = { 1, 0, -1, 0, 0 };
int Position::dy[] = { 0, 1, 0, -1, 0 };

std::string to_string(Direction dir) {
    auto it = dir_to_string_map.find(dir);
    Expects(it != std::end(dir_to_string_map));
    return it->second;
}

Direction from_string(const std::string& dir_str) {
    auto it = string_to_dir_map.find(dir_str);
    GSL_CONTRACT_CHECK("The given string doesn't match any direction.",
                       it != std::end(string_to_dir_map));
    return it->second;
}

Direction random_direction() {
    int random_ordinal = rl::util::random_in_range<int>(0, grid::DIR_COUNT);
    Direction d = directions[random_ordinal];
    return d;
}

} // namespace
