#pragma once

#include <vector>

#include "util/random.h"

namespace rl {

/**
 * DistributionList is a 1-D version of DistributionTree.
 *
 * Its simplicity allows it extra features such as the ability to be copy constructed.
 *
 */
template<typename T, typename Weight,
        // Only all the compilation of this class when Weight is a signed numeric type.
        typename = std::enable_if_t<std::is_signed<Weight>::value>>
class DistributionList {
public:

    class Entry {
    public:
        Entry() = default;
        Entry(const Entry&) = default;
        Entry(Entry&&) = default;
        Entry& operator=(const Entry&) = default;
        Entry& operator=(Entry&&) = default;

        Entry(Weight cumulative_begin, Weight weight, T* data) :
            cumulative_begin_(cumulative_begin),
            weight_(weight),
            data_(data)
        {}

        const T* data() const {
            return data_;
        }

        T* data() {
            return const_cast<T*>(static_cast<const Entry*>(this)->data());
        }

        Weight weight() const {
            return weight_;
        }

        Weight cumulative_begin() const {
            return cumulative_begin_;
        }

        Weight cumulative_end() const {
            return cumulative_begin_ + weight_;
        }

    private:
        Weight cumulative_begin_ = -1;
        Weight weight_ = -1;
        T* data_ = nullptr;
    };

    using Entries = std::vector<Entry>;

    void add(Weight weight, T* data) {
        Expects(weight > 0);
        Weight begin = total_weight();
        list_.emplace_back(begin, weight, data);
    }

    const T* random() const {
        Expects(!list_.empty());
        // Short-cut return if there is only one element.
        if(list_.size() == 1) {
            return list_.front().data();
        }
        Weight cumulative_pos = util::random::random_in_range<Weight>(0, total_weight());
        Ensures(cumulative_pos >= 0 and cumulative_pos < total_weight());
        std::size_t lower = 0;
        std::size_t upper = list_.size() -1;
        std::size_t mid;
        auto bisect_condition = [](Weight cumulative_end, Weight target) {
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

    T* random() {
        return const_cast<T*>(static_cast<const DistributionList*>(this)->random());
    }

    Weight total_weight() const {
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
