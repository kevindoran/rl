cmake_minimum_required(VERSION 3.10.2)
project(reinforcement)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wno-long-long -pedantic -Wno-unused-variable -DNDEBUG")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wno-long-long -pedantic -Wno-unused-variable -ftrapv")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# Conan
if(EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup()
else()
    message(WARNING "The file conanbuildinfo.cmake doesn't exist, you have to run conan install first")
endif()

set(3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty)

# Qt
set(CMAKE_AUTOMOC ON)
#set(CMAKE_AUTOUIC ON)
# Removed for the moment, as it isn't really used. Removing it saves a lot of time
# building via Travis CI.
#find_package(Qt5 REQUIRED Widgets)
find_package(GSL REQUIRED)


# Library
add_library(reinforcement
        src/rl/GridWorld.h
        src/grid/Grid.h
        src/grid/Grid.cpp
        src/rl/Policy.cpp
        src/rl/Policy.h
        src/rl/DistributionTree.h
        src/rl/DeterministicPolicy.h
        src/rl/ValueTable.h
        src/rl/IterativePolicyEvaluator.h
        src/util/DereferenceIterator.h
        src/util/RangeWrapper.h
        src/util/random.h
        src/rl/DeterministicImprover.h
        src/rl/StochasticPolicy.h
        src/rl/DistributionList.h
        src/rl/RandomPolicy.h
        src/rl/Environment.h
        src/rl/impl/Environment.h
        src/rl/FirstVisitMCValuePredictor.h
        src/rl/Trial.h
        src/rl/impl/PolicyEvaluator.h
        src/rl/FirstVisitMCActionValuePredictor.h
        src/rl/ActionValueTable.h
        src/rl/ActionValuePolicyImprover.h
        src/rl/MCEvaluator3.h
        src/rl/MCEvaluator3.cpp
        src/util/random.cpp
        src/rl/StateActionMap.h
        src/rl/QeGreedyPolicy.h
        src/rl/BlendedPolicy.h
        src/rl/TDEvaluator.cpp
        src/rl/TDEvaluator.h
        src/rl/SarsaImprover.h
        src/rl/impl/PolicyImprover.h
        src/rl/QLearningImprover.h
        src/rl/GradientMCLinear.h
        )
target_include_directories(reinforcement
        PUBLIC
        ${PROJECT_SOURCE_DIR}/src)
# 3rd party dependencies
# The add_compile_definitions() method is supported in cmake 3.12.
#add_compile_definitions(GSL_THROW_ON_CONTRACT_VIOLATION)
add_definitions(-DGSL_THROW_ON_CONTRACT_VIOLATION)
target_link_libraries(reinforcement ${CONAN_LIBS})

# Tests
add_executable(runTests
        test/distribution_tree.cpp
        test/grid_world.cpp
        test/range_wrapper.cpp
        test/iterative_policy_evaluation.cpp
        test/policy_iteration.cpp
        test/common/ExamplePolicies.h
        test/example_environments.cpp
        test/common/TestEnvironment.h
        test/distribution_list.cpp
        test/policy_evaluation.cpp
        test/common/PolicyEvaluationTests.h
        test/common/PolicyEvaluationTests.cpp
        test/policy.cpp
        test/common/suttonbarto/Exercise4_1.h
        test/common/suttonbarto/Exercise4_1.cpp
        test/common/suttonbarto/Exercise4_2.h
        test/common/suttonbarto/Exercise5_1.h
        test/common/suttonbarto/Example6_5.h
        test/common/suttonbarto/Example6_5.cpp
        test/qegreedy_policy.cpp
        test/common/suttonbarto/Example6_6.h
        test/common/suttonbarto/Example6_6.cpp
        test/common/suttonbarto/BlackjackEnvironment.h
        test/common/suttonbarto/CarRentalEnvironment.h
        test/common/suttonbarto/CliffWorld.h
        test/common/suttonbarto/WindyGridWorld.h
        test/common/suttonbarto/RandomWalk.h
        )

target_link_libraries(runTests gtest gtest_main)
target_link_libraries(runTests ${CONAN_LIBS})
target_link_libraries(runTests reinforcement)
target_include_directories(
        runTests
        PRIVATE
        ${GSL_INCLUDE_DIRS})
target_include_directories(
        runTests
        PRIVATE
        ${PROJECT_SOURCE_DIR}/test/common)
target_link_libraries(runTests GSL::gsl GSL::gslcblas)
enable_testing()
# Add a test to the project to be run by ctest
#add_test(<name> <executable>)
add_test(gtests runTests)


# Main
#add_executable(reinforcement_main src/main.cpp)
#target_link_libraries(reinforcement_main Qt5::Widgets)
#target_link_libraries(reinforcement_main reinforcement)
