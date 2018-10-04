#pragma once

#include <vector>
#include <gsl/gsl>
#include <stack>
#include <glog/logging.h>

#include "util/Random.h"

namespace rl {

template<typename T>
class DistributionTree {

public:
    explicit DistributionTree() = default;
    // Can't copy unless we manually treat copying of our root node.
    DistributionTree(const DistributionTree&) = delete;
    DistributionTree& operator=(const DistributionTree&) = delete;
    DistributionTree(DistributionTree&&) = default;
    DistributionTree& operator=(DistributionTree&&) = default;
    ~DistributionTree() = default;

    /**
     *
     * A node represents a contiguous region in a probability distribution.
     *
     * A node owns a segment of a 1-d cumulative distribution.
     * +----------------------
     * |      |  weight   |     ...
     * +----------------------
     *        ^           ^
     *        begin       end  (end = begin + weight)
     *
     * If a node covers a region beginning at zero with weight, w, then the region covered is
     * considered to be [0, weight).
     */
    class Node {
        friend class DistributionTree;
    public:
        using ID = int;

        explicit Node() = default;
        explicit Node(ID id, long weight, T* data) : id_(id), weight_(weight), data_(data) {}
        // We can't have default copy ctrs as we have a vector of unique pointers as a member, and
        // unique_ptr<T> is not copyable.
        // https://stackoverflow.com/questions/21943569/deleted-function-unique-ptr
        Node(const Node&) = delete;
        Node& operator=(const Node&) = delete;
        Node(Node&&) = default;
        Node& operator=(Node&&) = default;
        ~Node() = default;



        Node& add_child(long weight = 0, T* data = nullptr) {
            Ensures(children_.size() <= std::numeric_limits<ID>::max());
            ID id = static_cast<ID>(children_.size());
            return add_child_with_id(id, weight, data);
        }

        Node& add_child_with_id(ID id, long weight = 0, T* data = nullptr) {
            std::unique_ptr<Node>& added = children_.emplace_back(
                    std::make_unique<Node>(id, weight, data));
            auto res = id_to_node_.emplace(id, *added);
            bool inserted = res.second;
            // IDs should be unique.
            Ensures(inserted);
            return *added;
        }

        const Node& random_child() const {
            Expects(!children_.empty());
            long cumulative_pos = util::random_in_range(cumulative_begin_, cumulative_begin_ + weight_);
            return child_at_cumulative_pos(cumulative_pos);
        }

        Node& random_child() {
            return const_cast<Node&>(static_cast<const Node*>(this)->random_child());
        }

        Node& random_leaf() {
            if(children_.empty()) {
                return *this;
            }
            return random_child().random_leaf();
        }

        /**
         * This method is likely to have subtle bugs.
         * It should be switched to using the STL binary_search method. Keeping it here as it's
         * always nice to practice.
         */
        const Node& child_at_cumulative_pos(long cumulative_pos) const {
            Ensures(cumulative_pos < (cumulative_begin_ + weight_));
            Ensures(!children_.empty());
            std::size_t lower = 0;
            std::size_t upper = children_.size() - 1;
            std::size_t mid;
            auto bisect_condition = [](long cumulative_end, long target) {
                return target < cumulative_end;
            };
            while(lower < upper) {
                mid = lower + (upper - lower) / 2;
                if (bisect_condition(CHECK_NOTNULL(children_[mid])->cumulative_end(), cumulative_pos)) {
                    upper = mid;
                } else {
                    lower = mid + 1;
                }
            }
            Node& found = *CHECK_NOTNULL(children_[lower]);
            Ensures(found.cumulative_begin() <= cumulative_pos);
            long range_end = found.cumulative_begin() + found.weight();
            Ensures(range_end > cumulative_pos);
            return found;

        }

        long cumulative_begin() const {
            return cumulative_begin_;
        }

        long weight() const {
            return weight_;
        }

        long cumulative_end() const {
            return cumulative_begin_ + weight_;
        }

        T* data() {
            return data_;
        }

        const T* data() const {
            return data_;
        }

        void set_data(T* data) {
            data_ = data;
        }

        const Node& child(std::size_t index) const {
            Expects(index < children_.size());
            return *children_[index];
        }

        Node& child(std::size_t index) {
            // From Effictive C++.
            return const_cast<Node&>(static_cast<const Node*>(this)->child(index));
        }

        const Node& child_with_id(ID id) const {
            const auto it = id_to_node_.find(id);
            Expects(it != std::end(id_to_node_));
            return it->second;
        }

        bool has_child_with_id(ID id) const {
            bool found = (id_to_node_.count(id) > 0);
            return found;
        }

        Node& child_with_id(ID id) {
            return const_cast<Node&>(static_cast<const Node*>(this)->child_with_id(id));
        }

        std::size_t child_count() const {
            return children_.size();
        }

    private:
        // As we want clients to be able to store references/pointers to the nodes, we need to
        // place them on the heap (or use a container other that vector- vector can move its
        // elements in memory after an add).
        std::vector<std::unique_ptr<Node>> children_;
        std::unordered_map<ID, std::reference_wrapper<Node>> id_to_node_;
        // Uninitialized member variables will have garbage data. So initialize.
        int id_ = -1;
        long weight_ = -1;
        long cumulative_begin_= -1;
        T* data_ = nullptr;
    };

public:

    const Node& root_node() const {return root_node_;}

    Node& root_node() {
        return const_cast<Node&>(static_cast<const DistributionTree*>(this)->root_node());
    }

    void update_weights() {
        update_weights(root_node_, 0);
    }

    T& store() {
        return *data_storage_.emplace_back(std::make_unique<T>());
    }

    void dfs(std::function<void(const Node&)> to_run) const {
        dfs(to_run, root_node_);
    }

    void dfs(std::function<void(const Node&)> to_run, const Node& starting_from) const {
        std::stack<const Node*> stack;
        stack.emplace(&starting_from);
        while(!stack.empty()) {
            const Node& n = *CHECK_NOTNULL(stack.top());
            stack.pop();
            to_run(n);
            for(std::size_t i = 0; i < n.child_count(); i++) {
                stack.emplace(&n.child(i));
            }
        }
    }

private:
    Node root_node_;
    /**
     * Optional storage. Useful if the client doesn't want to manage the lifetime of the data held
     * by the nodes. This storage should place elemnets on the heap so that their positions don't
     * move.
     */
    std::vector<std::unique_ptr<T>> data_storage_;

    long update_weights(Node& n, long next_cumulative_start) {
        n.cumulative_begin_ = next_cumulative_start;
        // Non-leaf node.
        if(!n.children_.empty()) {
            // Reset branch weight to zero. It should be calculated from the children.
            n.weight_ = 0;
            for (std::unique_ptr<Node> &child : n.children_) {
                long child_weight = update_weights(*child, next_cumulative_start);
                next_cumulative_start += child_weight;
            }
            n.weight_ = next_cumulative_start - n.cumulative_begin_;
        }
        return n.weight_;
    }
};

} // namespace rl
