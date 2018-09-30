#include "gtest/gtest.h"
#include <vector>
#include "util/RangeWrapper.h"

struct Dummy {
    int value = 0;
};

TEST(RangeWrapperTest, loop_over_int_vector) {
    std::vector<Dummy> vec;
    const int count = 20;
    for(int i = 0; i < count; i++) {
        vec.emplace_back(Dummy{i});
    }
    using ItType = std::vector<Dummy>::const_iterator;
    rl::util::RangeWrapper<ItType> dummys(vec.begin(), vec.end());

    bool seen[count] = {false};
    for(int i = 0; i < count; i++) {
        ASSERT_FALSE(seen[count]) << "The test is broken if this fails.";
    }
    for(const Dummy& d : dummys) {
        seen[d.value] = true;
    }

    for(int i = 0; i < count; i++) {
        ASSERT_TRUE(seen[count]);
    }
}
