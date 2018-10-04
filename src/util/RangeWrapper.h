#pragma once

/**
 * Wraps two iterators so that the Wapper can be used in a for-each loop.
 */

namespace rl {
namespace util {

template<typename IteratorType>
class RangeWrapper {
public:
    RangeWrapper(IteratorType begin_it, IteratorType end_it) :
        begin_(begin_it), end_(end_it)
    {}

    IteratorType begin() {
        return begin_;
    }

    IteratorType end() {
        return end_;
    }

private:
    IteratorType begin_;
    IteratorType end_;
};

} // namespace util
} // namespace rl
