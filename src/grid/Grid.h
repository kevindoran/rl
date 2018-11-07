#pragma once

#include <string>
#include <unordered_map>
#include <gsl/gsl>
#include <vector>
#include <sstream>

namespace grid {

// Could use enum class here instead- more type safety but less convenient when wishing to treat
// the enum values as integers.
enum Direction {
    RIGHT = 0,
    DOWN = 1,
    LEFT = 2,
    UP = 3,
    NONE = 4  // Useful for return values as an optional alternative to throwing an exception.
};
static const int DIR_COUNT = 4;
static Direction directions[] = {Direction::RIGHT, Direction::DOWN, Direction::LEFT, Direction::UP};

/*
 * Until we have a better solution for enum serialization, we will do it like this:
 *
 * https://stackoverflow.com/questions/28828957/enum-to-string-in-modern-c11-c14-c17-and-future-c20
 *
 */
namespace {
const std::unordered_map<Direction, std::string> dir_to_string_map {
        {Direction::RIGHT, "right"},
        {Direction::DOWN,  "down"},
        {Direction::LEFT,  "left"},
        {Direction::UP,    "up"},
        {Direction::NONE,  "none"}
};

const std::unordered_map<std::string, Direction> string_to_dir_map {
        {"right", Direction::RIGHT},
        {"down",  Direction::DOWN},
        {"left",  Direction::LEFT},
        {"up",    Direction::UP},
        {"none",  Direction::NONE}
};
} // namespace

std::string to_string(Direction dir);

Direction from_string(const std::string& dir_str);

Direction random_direction();

/**
 * Represents a 2D coordinate (y, x) in ZxZ.
 */
class Position {
public:
    Position translate(Direction dir, int steps) const {
        return Position{y + dy[dir] * steps, x + dx[dir] * steps};
    }

    Position adj(Direction dir) const {
        return Position{y + dy[dir], x + dx[dir]};
    }

    bool operator==(const Position& other) const {
        return y == other.y && x == other.x;
    }

    friend std::ostream &operator<<(std::ostream &os, const Position &pos) {
        os << "(" << pos.y << "," << pos.x << ")";
        return os;
    }

    std::string to_string() {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

public:
    int y;
    int x;
    // Map from Direction to delta.
    // Example: change in x coordinate when moving in Direction d = dx[d].
    static int dx[];
    static int dy[];
};


template<int HEIGHT, int WIDTH>
class Grid {
    static_assert(WIDTH > 0);
    static_assert(HEIGHT > 0);

public:
    static const int INVALID_TILE = -1;
    static const int TILE_COUNT = WIDTH * HEIGHT;

    static int width() {return WIDTH; }
    static int height() {return HEIGHT; }

    Grid() {
        for(int i = 0; i < TILE_COUNT; i++) {
            pos_map[i] = Position{i / WIDTH, i % WIDTH};
        }
    }

    /**
     * Returns the Position corresponding to the given tile.
     *
     * Example, tile_id of 5 corresponds to (y=1, x=1) when WIDTH = 4.
     *
     * Currently, this returns a reference to a static pre-calculated Position.
     * Optionally, this could be returned by value. My guess is that by reference allows slightly
     * faster comparisons and slightly lower memory footprint. It's worth profiling with different
     * optimization levels.
     */
    inline const Position& to_position(int tile_id) const {
        Expects(tile_id >= 0);
        Expects(tile_id < TILE_COUNT);
        return pos_map[tile_id];
    }

    /**
     * Calculates and returns a tile a certain number of spaces away in a specific direction.
     *
     * \param from   from this tile.
     * \param dir    in this direction.
     * \param steps  this number of steps away.
     * \return       the id of the tile \c steps number of steps away from \from in direction \dir.
     *               If no tile exists then \c INVALID_TILE is returned.
     */
    static int tile_at(const Position& from, Direction dir, int steps) {
        int x = from.x + steps * Position::dx[dir];
        int y = from.y + steps * Position::dy[dir];
        if(x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) {
            return INVALID_TILE;
        } else {
            return x + y*WIDTH;
        }
    }

    /**
     * Calculates and returns an adjacent tile.
     *
     * Note: no bounds checking is performed by this method.
     *
     * \param tile  from this tile.
     * \param dir   in this direction.
     * \return      the id of the tile 1 space from \c tile in direction \c dir.
     */
    static int adj_tile(int tile, Direction dir)  {
        return tile + (Position::dx[dir] + WIDTH * Position::dy[dir]);
    }

    /**
     * Loops a position back into the grid if it is outside the grid boundary.
     *
     * For example, in a 2x2 grid, (0, 3) will produce (0, 1) with this method.
     *
     * \param p    the position to have looped.
     * \return     a Position within the grid. Hm... a bit of inconsistency here. Returning by value
     *             instead of by const ref.
     */
    static const Position modulo(const Position& p) {
        return Position{p.y % HEIGHT, p.x % WIDTH};
    }

    /**
     * Determines if the given position exists on this grid.
     *
     * \return \c true if \c p is a valid position on the grid, \c false otherwise.
     */
    static bool is_valid(const Position& p) {
        return p.y >= 0 && p.y < HEIGHT && p.x >= 0 && p.x < WIDTH;
    }

    static inline int dist(const Position& a, const Position& b) {
        return abs(a.x - b.x) + abs(a.y - b.y);
    }

    static inline int dist(int a, int b) {
        return abs(a % WIDTH - b % WIDTH) + abs(a / WIDTH - b / WIDTH);
    }

    static inline int to_id(const Position& pos) {
        return pos.x + WIDTH * pos.y;
    }

    static inline int to_id(int y, int x) {
        return x + WIDTH * y;
    }

    /**
     * Calculates and returns the neighbouring tiles of a given tile.
     *
     * Neighbours are considered to be the tiles above, below, left and right of a given tile.
     * Tiles diagonally next to the given tile are not included.
     *
     * Note: we could optionally pre-compute these neighbours for a whole grid, similar to our
     * usage of pos_map. It would take a bit more than width * height * 4 * sizeof(int).
     *
     * \param t   get the neighbours of this tile.
     * \return    the tile's neighbours: a \c vector of between 2 to 4 tile ids.
     */
    inline std::vector<int> neighbours(int t) const {
        std::vector<int> ans;
        if(t % WIDTH != 0) {
            ans.push_back(t - 1);
        }
        if(t % WIDTH != WIDTH - 1) {
            ans.push_back(t + 1);
        }
        if(t >= WIDTH) {
            ans.push_back(t - WIDTH);
        }
        if(t < WIDTH * (HEIGHT - 1)) {
            ans.push_back(t + WIDTH);
        }
        return ans;
    }

    /**
     * Calculates the neighbours and neighbour count for a given tile. Returns via output param.
     *
     * \param t            get the neighbours of this tile.
     * \param ans[out]     an array where the neighbours will be stored. Must be length 4 or more.
     * \param count[out]   the number of neigbours.
     */
    void neighbours(int t, int* ans, int* count) const {
        (*count) = 0;
        if(t % WIDTH != 0) {
            ans[(*count)++] = (t - 1);
        }
        if(t % WIDTH != WIDTH - 1) {
            ans[(*count)++] = (t + 1);
        }
        if(t >= WIDTH) {
            ans[(*count)++] = (t - WIDTH);
        }
        if(t < WIDTH * (HEIGHT - 1)) {
            ans[(*count)++] = (t + WIDTH);
        }
    }

    // Not sure why this was needed.
    std::vector<int> neighbours_incl(int t) const {
        std::vector<int> ans;
        ans.push_back(t);
        if(t % WIDTH != 0) {
            ans.push_back(t - 1);
        }
        if(t % WIDTH != WIDTH - 1) {
            ans.push_back(t + 1);
        }
        if(t >= WIDTH) {
            ans.push_back(t - WIDTH);
        }
        if(t < WIDTH * (HEIGHT - 1)) {
            ans.push_back(t + WIDTH);
        }
        return ans;
    }

    inline std::vector<Position> neighbours(const Position& pos) const {
        std::vector<Position> ans;
        if(pos.y > 0) {
            ans.push_back(Position{pos.y-1, pos.x});
        }
        if(pos. y < HEIGHT - 1) {
            ans.push_back(Position{pos.y+1, pos.x});
        }
        if(pos.x > 0) {
            ans.push_back(Position{pos.y, pos.x - 1});
        }
        if(pos.x < WIDTH - 1) {
            ans.push_back(Position{pos.y, pos.x + 1});
        }
        return ans;
    }

private:
    Position pos_map[TILE_COUNT];

    // We could have an optional per-tile data member also. std::enable_if could remove it.
};

} // namespace grid

