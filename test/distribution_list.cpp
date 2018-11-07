#include "gtest/gtest.h"
#include <glog/logging.h>
#include "gsl/gsl_cdf.h"
#include <list>

#include "rl/DistributionList.h"

template<typename NumberType>
class DistributionListTypeF : public ::testing::Test {

};

using NumberTypes = ::testing::Types<int, long, float, double, long long, long double>;
TYPED_TEST_CASE(DistributionListTypeF, NumberTypes);
struct DummyStruct {};

namespace {
template<typename NUM>
typename std::enable_if_t<std::is_integral<NUM>::value, std::vector<NUM>>
get_test_numbers() {
    return std::vector<NUM>{1, 2, 3, 4, 5, 60, 700, 8000};
}

// Alternatively, the enable_if can be placed in the template
// template<typename NUM, typename std::enable_if_t<std::is_floating_point<NUM>::value, int> = 0>
template<typename NUM>
typename std::enable_if_t<std::is_floating_point<NUM>::value, std::vector<NUM>>
get_test_numbers() {
    return std::vector<NUM>{0.00001, 0.002, 0.03, 0.4, 0.9999, 0.1,
                            1.0, 1.001, 2.0, 5, 60, 700, 1000.1 };
}
} // namespace

/**
 * Tests constructing a DistributionList::Entry and calling getter methods.
 *
 * The test carries out simple construction and getter calls for a number of different Entry type
 * instances (varying the Weighting template parameter).
 */
TYPED_TEST(DistributionListTypeF, Entry_constructor_and_getters) {
    // Setup.
    using EntryType = typename rl::DistributionList<DummyStruct, TypeParam>::Entry;
    using NumType = TypeParam;
    // TODO: make it clear what the precision guarantees are.
    // We are getting away with using a precision = epsilon as our test code has the same
    // calculation for each value. Actually precision = 0 would do fine. If
    // (instead weight = cumulative_end - cumulative_begin) in the test code, we would need broader
    // bounds.
    NumType precision = std::numeric_limits<NumType>::epsilon();
    DummyStruct d;

    // Test
    for(NumType cumulative_begin : get_test_numbers<NumType>()) {
        for(NumType weight : get_test_numbers<NumType>()) {
            EntryType entry(cumulative_begin, weight, &d);
            EXPECT_NEAR(weight, entry.weight(), precision);
            EXPECT_NEAR(cumulative_begin, entry.cumulative_begin(), precision);
            NumType cumulative_end(cumulative_begin + weight);
            EXPECT_NEAR(cumulative_end, entry.cumulative_end(), precision);
            EXPECT_EQ(&d, entry.data());
        }
    }
}

/**
 * Tests the total_weight() method.
 *
 * Tests that:
 *    1. total_weight() is zero if there are no entries.
 *    2. total_weight() calculates and returns the correct value.
 */
TYPED_TEST(DistributionListTypeF, total_weight) {
    // Setup
    using NumType = TypeParam;
    rl::DistributionList<DummyStruct, NumType> dist_list;

    // Test
    // 2. Empty case.
    ASSERT_EQ(0, dist_list.total_weight());

    // 2. total_weight() calculates the correct value.
    NumType sum = 0;
    for(NumType weight : get_test_numbers<NumType>()) {
        dist_list.add(weight, nullptr);
        sum += weight;
    }
    // Another precarious number comparison. If the test calculations differ from those used in
    // the implementation, then this comparison will need bounds.
    ASSERT_EQ(sum, dist_list.total_weight());
}

/**
 * Tests the add() method.
 *
 * Tests that:
 *    1. An entry can't be added with zero or negative weight.
 */
TYPED_TEST(DistributionListTypeF, add) {
    // Setup
    using NumType = TypeParam;
    rl::DistributionList<DummyStruct, NumType> dist_list;

    // Test
    ASSERT_ANY_THROW(dist_list.add(-1, nullptr));
    ASSERT_ANY_THROW(dist_list.add(0, nullptr));
}

namespace {

template<typename Weight>
class Counter {
public:
    Counter(Weight weight) : weight_(weight) {}

    int increment() {
        return ++count_;
    }
    int value() const {return count_;}

    Weight weight() const {return weight_;}

private:
    int count_ = 0;
    Weight weight_;
};

template<typename NUM>
typename std::enable_if_t<std::is_integral<NUM>::value, std::vector<NUM>>
get_test_weightings() {
    return std::vector<NUM>{1, 1, 1, 1, 2, 2, 2, 5, 10}; // Total weight: 25.
}

// Alternatively, the enable_if can be placed in the template
// template<typename NUM, typename std::enable_if_t<std::is_floating_point<NUM>::value, int> = 0>
template<typename NUM>
typename std::enable_if_t<std::is_floating_point<NUM>::value, std::vector<NUM>>
get_test_weightings() {
    return std::vector<NUM>{0.1, 0.4, 0.5, 1.0, 1.0, 2.0, 2.5, 2.5, 5, 10}; // Total weight: 25
}

} // namespace

/**
 * Tests the random method.
 *
 * Tests that:
 *    1. The results follow the expected distribution.
 *    2. An exception is thrown if there are no entries.
 */
TYPED_TEST(DistributionListTypeF, random) {
    // Setup
    using NumType = TypeParam;
    const int samples = 100000;
    const int total_weight = 25;
    const int samples_per_unit_weight = samples / total_weight;
    const double confidence_required = 0.98;
    ASSERT_TRUE(samples % total_weight == 0) << "The test is broken if this fails.";
    std::list<Counter<NumType>> counters;
    rl::DistributionList<Counter<NumType>, NumType> counter_list;
    double actual_total_weight = 0;
    for(NumType weight : get_test_weightings<NumType>()) {
        counters.emplace_back(weight);
        counter_list.add(weight, &counters.back());
        actual_total_weight += weight;
    }
    ASSERT_NEAR(total_weight, actual_total_weight, 1e-8) << "The test is broken if this fails.";

    for(int i = 0; i < samples; i++) {
        Counter<NumType>& data = *CHECK_NOTNULL(counter_list.random());
        data.increment();
    }

    // Test
    // 1. Check the samples against the expected distribution.
    // Chi-squared test.
    // Following steps outlined at: https://stattrek.com/chi-square-test/goodness-of-fit.aspx
    // Using the method: X^2 = ( (O-E)^2 / E )
    double X2 = 0;
    for(auto& counter : counters) {
        double expected = samples_per_unit_weight * counter.weight();
        double observed = counter.value();
        X2 += std::pow(observed - expected, 2) / expected;
    }
    const int degrees_of_freedom = counter_list.entries().size() - 1;
    const double p_value = 1 - gsl_cdf_chisq_P(X2, degrees_of_freedom);
    const double cut_off = 1 - confidence_required;
    ASSERT_GT(p_value, cut_off);

    // Test
    // 2.
    rl::DistributionList<DummyStruct, NumType> dist_list;
    ASSERT_ANY_THROW(dist_list.random());
}
