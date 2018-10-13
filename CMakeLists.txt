cmake_minimum_required(VERSION 3.10.2)
project(reinforcement)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-unused-variable")
set(CMAKE_BINARY_DIR  ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(3RD_PARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty)

# Qt
set(CMAKE_AUTOMOC ON)
#set(CMAKE_AUTOUIC ON)
find_package(Qt5 REQUIRED Widgets)
#find_package(glog 0.3.5 REQUIRED)
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
        src/rl/PolicyRunner.h
        src/rl/ValueFunction.h
        src/rl/IterativePolicyEvaluation.h
        src/util/DereferenceIterator.h
        src/util/RangeWrapper.h
        src/util/Random.h
        src/rl/PolicyIteration.h
        src/rl/StochasticPolicy.h
        src/rl/DistributionList.h
        src/rl/RandomGridPolicy.h
        src/rl/Environment.h
        src/rl/impl/Environment.h
        src/rl/FirstVisitMCValuePrediction.h
        src/rl/Trial.h)
target_include_directories(reinforcement
        PUBLIC
        ${PROJECT_SOURCE_DIR}/src)
# 3rd party dependencies
target_include_directories(reinforcement
        PUBLIC
        ${3RD_PARTY_DIR}/coreguidelines/include)
add_subdirectory(3rdparty/glog)
add_compile_definitions(GSL_THROW_ON_CONTRACT_VIOLATION)
target_link_libraries(reinforcement Qt5::Widgets)
target_link_libraries(reinforcement glog)
# Use the following if using the find_package(glog 0.3.5 REQUIRED)
#target_link_libraries(reinforcement glog::glog)

# Tests
# https://stackoverflow.com/a/21479008
add_subdirectory(3rdparty/googletest)
add_executable(runTests
        test/distribution_tree.cpp
        test/grid_world.cpp
        test/range_wrapper.cpp
        test/iterative_policy_evaluation.cpp
        test/policy_iteration.cpp
        test/common/ExamplePolicies.h)
target_link_libraries(runTests gtest gtest_main)
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
# Why doesn't this work?
#target_include_directories(runTests
#        PUBLIC
#        ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(
        ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
# Enable cmake testing
enable_testing()
# Add a test to the project to be run by ctest
#add_test(<name> <executable>)
add_test(gtests runTests)


# Main
add_executable(reinforcement_main src/main.cpp)
target_link_libraries(reinforcement_main reinforcement)
