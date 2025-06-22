cmake_minimum_required(VERSION 3.14)
project(embee VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(EMBEE_BUILD_EXAMPLES "Build example applications" ON)
option(EMBEE_BUILD_TESTS "Build tests" ON)
option(EMBEE_USE_OPENMP "Use OpenMP for parallelization" ON)

# Check for OpenMP
if(EMBEE_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# Add compiler optimizations for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Platform-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
    endif()
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

# Include directories
include_directories(include)

# Main library source files
set(EMBEE_SRC
    src/engine.cpp
    src/model.cpp
    src/tokenizer.cpp
    src/bpe_tokenizer.cpp
    src/sentencepiece_tokenizer.cpp
    src/tensor.cpp
    src/amb_format.cpp
    src/gguf_loader.cpp
    src/onnx_loader.cpp
)

# Main library
add_library(embee ${EMBEE_SRC})
target_include_directories(embee PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Examples
if(EMBEE_BUILD_EXAMPLES)
    add_executable(chat_cli examples/chat_cli.cpp)
    target_link_libraries(chat_cli PRIVATE embee)
    
    add_executable(benchmark examples/benchmark.cpp)
    target_link_libraries(benchmark PRIVATE embee)
endif()

# Tests
if(EMBEE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS embee
    EXPORT embee-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
)

# Export targets
install(EXPORT embee-targets
    FILE embee-targets.cmake
    NAMESPACE embee::
    DESTINATION lib/cmake/embee
)