cmake_minimum_required(VERSION 3.20)
project(wayy_db VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(WAYY_BUILD_PYTHON "Build Python bindings" ON)
option(WAYY_BUILD_TESTS "Build unit tests" ON)
option(WAYY_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(WAYY_USE_AVX2 "Enable AVX2 SIMD optimizations" ON)
option(WAYY_USE_LZ4 "Enable LZ4 compression" OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(WAYY_USE_AVX2)
        add_compile_options(-mavx2 -mfma)
    endif()
endif()

# Core library
add_library(wayy_core STATIC
    src/types.cpp
    src/column.cpp
    src/table.cpp
    src/database.cpp
    src/mmap_file.cpp
    src/ops/aggregations.cpp
    src/ops/joins.cpp
    src/ops/window.cpp
)

target_include_directories(wayy_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Need PIC for linking into shared libraries (Python module)
set_target_properties(wayy_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(WAYY_USE_AVX2)
    target_compile_definitions(wayy_core PUBLIC WAYY_USE_AVX2=1)
endif()

if(WAYY_USE_LZ4)
    find_package(lz4 REQUIRED)
    target_link_libraries(wayy_core PRIVATE lz4::lz4)
    target_compile_definitions(wayy_core PUBLIC WAYY_USE_LZ4=1)
endif()

# Python bindings
if(WAYY_BUILD_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

    # Fetch pybind11 (v2.13+ required for free-threaded Python support)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(_core python/bindings.cpp)
    target_link_libraries(_core PRIVATE wayy_core)

    # Install Python module to the package directory
    # scikit-build-core will place this in the wayy_db package
    install(TARGETS _core DESTINATION wayy_db COMPONENT python)
endif()

# Tests
if(WAYY_BUILD_TESTS)
    enable_testing()

    # Fetch GoogleTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # Prevent overriding parent project's compiler/linker settings (Windows)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(wayy_tests
        tests/test_types.cpp
        tests/test_column.cpp
        tests/test_table.cpp
        tests/test_mmap.cpp
        tests/test_joins.cpp
    )

    target_link_libraries(wayy_tests PRIVATE
        wayy_core
        GTest::gtest
        GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(wayy_tests)
endif()

# Benchmarks
if(WAYY_BUILD_BENCHMARKS)
    find_package(benchmark REQUIRED)

    add_executable(wayy_benchmarks
        benchmarks/bench_aggregations.cpp
        benchmarks/bench_joins.cpp
    )

    target_link_libraries(wayy_benchmarks PRIVATE
        wayy_core
        benchmark::benchmark
    )
endif()
