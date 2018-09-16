cmake_minimum_required(VERSION 3.10.2)
project(reinforcement)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BINARY_DIR  ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# Qt
set(CMAKE_AUTOMOC ON)
#set(CMAKE_AUTOUIC ON)
find_package(Qt5 REQUIRED Widgets)


add_executable(reinforcement src/main.cpp)

target_include_directories(reinforcement
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(reinforcement Qt5::Widgets)