project(unit_tests)

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)

add_executable(linear-algebra-library-ut main.cpp Array.cpp math.cpp common.hpp Memory.cpp MemoryView.cpp)
target_include_directories(linear-algebra-library-ut PRIVATE
        googletest/googletest/include googletest/googlemock/include
        ${CMAKE_SOURCE_DIR}/src
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_link_libraries(linear-algebra-library-ut gmock gtest linear-algebra-library)