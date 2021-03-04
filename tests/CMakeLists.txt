project(unit_tests)

set(CMAKE_CXX_FLAGS "${DEFAULT_CXX_FLAGS}") #disable warnings when compiling gtest

add_subdirectory(googletest)

add_executable(unit_tests main.cpp Array.cpp math.cpp common.hpp Memory.cpp MemoryView.cpp)

target_include_directories(unit_tests PRIVATE googletest/googletest/include googletest/googlemock/include ${CMAKE_SOURCE_DIR}/src "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_link_libraries(unit_tests gmock gtest test_lib)