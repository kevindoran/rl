#pragma once

#include <vector>

#include "util/Random.h"

namespace rl {

/**
 * DistributionList is a 1-D version of DistributionTree.
 *
 * Its simplicity allows it extra features such as the ability to be copy constructed.
 *
 */
template<typename T>
class DistributionList {
public:

    class Entry {
    public:
        Entry() = default;
        Entry(const Entry&) = default;
        Entry(Entry&&) = default;
        Entry& operator=(const Entry&) = default;
        Entry& operator=(Entry&&) = default;

        Entry(long cumulative_begin, long weight, T* data) :
            cumulative_begin_(cumulative_begin),
            cumulative_end_(cumulative_begin + weight),
            data_(data)
        {}

        const T* data() const {
            return data_;
        }

        long weight() const {
            return cumulative_end_ - cumulative_begin_;
        }

        long cumulative_begin() const {
            return cumulative_begin_;
        }

        long cumulative_end() const {
            return cumulative_end_;
        }

    private:
        long cumulative_begin_ = -1;
        long cumulative_end_ = -1;
        T* data_ = nullptr;
    };

    using Entries = std::vector<Entry>;

    void add(long weight, T* data) {
        long begin = total_weight();
        list_.emplace_back(begin, weight, data);
    }

    const T* random() const {
        Expects(!list_.empty());
        // Short-cut return if there is only one element.
        if(list_.size() == 1) {
            return list_.front().data();
        }
        long cumulative_pos = util::random_in_range(0l, total_weight());
        Ensures(cumulative_pos >= 0 and cumulative_pos < total_weight());
        std::size_t lower = 0;
        std::size_t upper = list_.size() -1;
        std::size_t mid;
        auto bisect_condition = [](long cumulative_end, long target) {
            return target < cumulative_end;
        };
        while(lower < upper) {
            mid = lower + (upper - lower) / 2;
            if(bisect_condition(list_[mid].cumulative_end(), cumulative_pos)) {
                upper = mid;
            } else {
                lower = mid + 1;
            }
        }
        Ensures(list_[lower].cumulative_begin() <= cumulative_pos);
        Ensures(list_[lower].cumulative_end() > cumulative_pos);
        return list_[lower].data();
    }

    long total_weight() const {
        if(list_.empty()) {
            return 0;
        }
        return list_.back().cumulative_end();
    }

    const Entries& entries() const {
        return list_;
    }

private:
    Entries list_;
};

} // namespace rl
