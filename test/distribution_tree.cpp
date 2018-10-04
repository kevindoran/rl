#include "gtest/gtest.h"
#include <limits>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include "gsl/gsl_randist.h"
#include "gsl/gsl_cdf.h"

#include "rl/DistributionTree.h"


/**
 * Things to test:
 *   * single layer
 *   * multi-layer
 *   * weights affect selection as expected (for random selection).
 *   * cumulative position is correct.
 *   * weight of parent is sum of child weights
 *
 */

using namespace rl;

namespace {
    class Counter {
    public:
        int increment() {
            //LOG(INFO) << "Incrementing. New val: " << count_ + 1;
            return ++count_;
        }
        int value() const {return count_;}

    private:
        int count_ = 0;
    };
    using CounterTree = DistributionTree<Counter>;
    using CounterNode = CounterTree::Node;


    void create_uniform_subtree(int levels, int children_per_level,
            int weight, CounterNode& start_node, CounterTree& tree) {
        Expects(levels > 0);
        Expects(children_per_level > 0);
        Expects(weight > 0);
        for(int i = 0; i < children_per_level; i++) {
            CounterNode& n = start_node.add_child(weight);
            if(levels == 1) {
                n.set_data(&tree.store());
            } else {
                create_uniform_subtree(levels-1, children_per_level, weight, n, tree);
            }
        }
    }

    CounterTree uniform_tree(int levels, int children_per_level, int weight=1) {
        Expects(levels > 0);
        Expects(children_per_level > 0);
        Expects(weight > 0);
        CounterTree tree;
        create_uniform_subtree(levels, children_per_level, weight, tree.root_node(), tree);
        tree.update_weights();
        // This will use a move ctr, so be careful with storing pointers to non-heap tree members.
        return tree;
    }
/*
    std::pair<double, double> binom_confidence_bounds(int trials, int success,
            double confidence) {
        Expects(confidence >= 0 and confidence <= 1.0);
        Expects(success >= 0 and success <= trials);
        double quantile = (1.0 - confidence) / 2.0;
        double lo = gsl_ran_beta_pdf(quantile, success, trials - success + 1);
        double hi = gsl_ran_beta_pdf(1-quantile, success + 1, trials - success);
        return std::make_pair(lo, hi);
    }

    double inverse_binom_cdf(int trials, double p, double quartile) {
        Expects(trials > 0);
        Expects(quartile >= 0 and quartile <= 1.0);
        double sum = 0;
        int x = 0;
        while(true) {
            sum += gsl_ran_binomial_pdf(x, p, trials);
            if(sum >= quartile) {
                return x;
            }
            x++;
        }
    }
*/


} // namespace



/**
 * All distributions should have a root node when created.
 */
TEST(DistributionTreeTest, test_root_node) {
   CounterTree dist_tree;
   ASSERT_NO_THROW(dist_tree.root_node());
}

/**
 * Tests that:
 *   1. The child count is correct.
 *   2. The child is at the expected index, with the expected weight and data.
 *   3. We can depend on the stability of the returned reference from add_child().
 *   4. A child can have a child.
 *   5. The child_count() is only counting direct children (not children of children).
 */
TEST(DistributionTreeTest, test_add_child) {
    // Setup.
    CounterTree dist_tree;
    Counter c;

    // Test.
    CounterNode& child1 = dist_tree.root_node().add_child();
    CounterNode& child2 = dist_tree.root_node().add_child(10, &c);
    // 1. The child count is correct.
    ASSERT_EQ(2, dist_tree.root_node().child_count());
    // 2. Child is at the expected index.
    ASSERT_EQ(&child1, &dist_tree.root_node().child(0));
    ASSERT_EQ(&child2, &dist_tree.root_node().child(1));
    // With the expected weight.
    ASSERT_EQ(0, child1.weight());
    ASSERT_EQ(10, child2.weight());
    // With the expected data.
    const Counter* child1_data = child1.data();
    ASSERT_EQ(nullptr, child1_data);
    ASSERT_EQ(&c, child2.data());
    // 3. The reference is stable. Add a few more children and make sure child 1 & 2 haven't moved.
    for(int i = 0; i < 10; i++) {
        dist_tree.root_node().add_child();
    }
    ASSERT_EQ(&child1, &dist_tree.root_node().child(0));
    ASSERT_EQ(&child2, &dist_tree.root_node().child(1));

    // 4. A child can have a child.
    std::size_t root_child_count_before = dist_tree.root_node().child_count();
    child1.add_child();
    // 5. child_count() of root_node isn't effected.
    ASSERT_EQ(root_child_count_before, dist_tree.root_node().child_count());
}

/**
 * Tests that children can be obtained as expected via their cumulative_pos.
 *
 * Tests that:
 *   1. Children can be queried their start of their cumulative range.
 *   2. Children can be queried by a point within their cumulative range.
 *   Note: for 1 & 2, we test on four sizes of child containers: 1, 2, 4 and 5.
 */
TEST(DistributionTreeTest, test_child_at_cumulative_pos) {
    // Setup.
    std::vector<int> sizes{1, 2, 4, 5};
    const int weight = 3;
    const int level_count = 1;
    std::vector<CounterTree> trees;
    for(int size : sizes) {
        trees.emplace_back(uniform_tree(level_count, size, weight));
    }

    // Test.
    // 1. Query via the beginning of a child's cumulative range.
    for(CounterTree& tree : trees) {
        CounterNode& root = tree.root_node();
        for(std::size_t i = 0; i < root.child_count(); i++) {
            long query_point = i*weight;
            ASSERT_EQ(&root.child(i), &root.child_at_cumulative_pos(query_point))
            << "Didn't find child (" << i << ") on node with (" << root.child_count() << ") children";
        }
    }

    // 2. Query at a non-endpoint of a child's cumulative range.
    for(CounterTree& tree : trees) {
        CounterNode& root = tree.root_node();
        for(std::size_t i = 0; i < root.child_count(); i++) {
            long query_point = i*weight + 1;
            ASSERT_EQ(&root.child(i), &root.child_at_cumulative_pos(query_point))
            << "Didn't find child (" << i << ") on node with (" << root.child_count() << ") children";
        }
    }

}

TEST(DistributionTreeTest, test_random_child) {
    // Setup.
    // 3 levels with 4 children per node.
    // l1: 4 nodes
    // l2: 16 nodes
    // l3: 64 nodes
    const int levels = 3;
    const int children_per_node = 4;
    const int leaf_count = std::pow(children_per_node, levels);
    const int trials_per_leaf = 100;
    const int trial_count = leaf_count * trials_per_leaf;
    const double confidence_required = 0.95;

    CounterTree tree{uniform_tree(levels, children_per_node)};
    // Randomly increment leaf counters.
    for(int t = 0; t < trial_count; t++) {
        CounterNode& l = tree.root_node().random_leaf();
        l.data()->increment();
    }
    // Collect all the leaves.
    std::vector<std::reference_wrapper<const CounterNode>> leaves;
    std::function<void(const CounterNode&)> fctn =
        [&leaves](const CounterNode& n){
            if(!n.child_count()) {
                leaves.emplace_back(n);
            }
        };
    tree.dfs(fctn);
    // Chi-squared test.
    // Using the method: X^2 = ( (O-E)^2 / E )
    double X2 = 0;
    for(std::reference_wrapper<const CounterNode> leaf : leaves) {
        int count = leaf.get().data()->value();
        const int observed = count;
        const int expected = trials_per_leaf;
        X2 += std::pow(observed - expected, 2) / expected;
    }
    const int degrees_of_freedom = leaf_count - 1;
    const double p_value = 1 - gsl_cdf_chisq_P(X2, degrees_of_freedom);
    const double cut_off = 1 - confidence_required;
    ASSERT_GT(p_value, cut_off);
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    //google::InitGoogleLogging(argv[0]);
    return RUN_ALL_TESTS();
}