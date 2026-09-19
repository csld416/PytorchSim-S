cmake_minimum_required(VERSION 3.15.0)
set(project_name "AiFrameworkSim")
project(${project_name})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include("${CMAKE_SOURCE_DIR}/build/conanbuildinfo.cmake")
conan_basic_setup()

# find_package(Boost 1.70 REQUIRED COMPONENTS program_options)
# message("LIB ${Boost_LIBRARY_DIRS}")
option(USE_RAMULATOR "USE_RAMULATOR" ON)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/extern)

# C++ setttings
set(CMAKE_CXX_STANDARD 17)
set(ONNX_ML 1)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/bin")
set(LIBRARY_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/build/lib")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
message("BINARY DIR ${CMAKE_BINARY_DIR}")
#if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#    add_compile_options(-fsanitize=address)
#    add_link_options(-fsanitize=address)
#endif()
# Build source
add_subdirectory("${PROJECT_SOURCE_DIR}/src")

# Add libaray ramulator
include_directories("${PROJECT_SOURCE_DIR}/include")
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/ramulator2")
include_directories("${PROJECT_SOURCE_DIR}/extern/ramulator2/src")
include_directories("${PROJECT_SOURCE_DIR}/extern/ramulator2/resources/wrappers")

# Add libaray booksim
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/booksim")

# Add libary protobuf
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/protobuf/cmake" EXCLUDE_FROM_ALL)
set_target_properties(libprotoc PROPERTIES FOLDER "external/protobuf")
set_target_properties(protoc PROPERTIES FOLDER "external/protobuf")

# Add libaray stonne core
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/stonneCore")

# Add libaray onnx
add_definitions("-DONNX_NAMESPACE=onnx")
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/onnx" EXCLUDE_FROM_ALL)
set_target_properties(onnx PROPERTIES FOLDER "extern/onnx")
set_target_properties(onnx_proto PROPERTIES FOLDER "extern/onnx")

target_include_directories(Simulator PUBLIC ${PROJECT_SOURCE_DIR}/extern/stonneCore/include)
target_include_directories(Simulator PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(Simulator PUBLIC ${PROJECT_SOURCE_DIR}/include/scheduler)
target_include_directories(Simulator PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(Simulator PUBLIC ${ONNX_INCLUDE_DIRS})
target_link_libraries(Simulator booksim2 ramulator sstStonne)
target_link_libraries(Simulator ${PROTOBUF_LIB} onnx_proto sstStonne ${CONAN_LIBS} stdc++fs)

# LegoSim / interchiplet integration: lets DMA.cc query a live SSD/DRAM
# latency simlet through the interchiplet sync protocol instead of replaying
# a pre-recorded trace.
if(NOT DEFINED ENV{SIMULATOR_ROOT})
    set(LEGOSIM_ROOT "/workspace/legomerged/eclab_legosim")
else()
    set(LEGOSIM_ROOT $ENV{SIMULATOR_ROOT})
endif()
if(NOT DEFINED ENV{NUSSD_PROTOCOL_DIR})
    set(NUSSD_PROTOCOL_DIR "${PROJECT_SOURCE_DIR}/../../protocol")
else()
    set(NUSSD_PROTOCOL_DIR $ENV{NUSSD_PROTOCOL_DIR})
endif()
target_include_directories(Simulator PUBLIC ${LEGOSIM_ROOT}/interchiplet/includes)
target_include_directories(Simulator PUBLIC ${PROJECT_SOURCE_DIR}/legosim)
target_include_directories(Simulator PUBLIC ${NUSSD_PROTOCOL_DIR})
target_link_libraries(Simulator ${LEGOSIM_ROOT}/interchiplet/lib/libinterchiplet_c.a pthread)

# SSD/DRAM latency simlet -- standalone binary, only needs the (header-only)
# interchiplet protocol, none of TOGSim's own dependencies.
add_executable(ssd_simlet "${PROJECT_SOURCE_DIR}/legosim/ssd_simlet.cpp")
target_include_directories(ssd_simlet PUBLIC ${LEGOSIM_ROOT}/interchiplet/includes)
target_link_libraries(ssd_simlet ${LEGOSIM_ROOT}/interchiplet/lib/libinterchiplet_c.a pthread)

# DRAM+interconnect latency simlet -- catch-all counterpart to ssd_simlet
# covering every DMA access (not just weight reads), standing in for
# TOGSim's real Dram/Interconnect models. Same standalone build shape.
add_executable(dram_simlet "${PROJECT_SOURCE_DIR}/legosim/dram_simlet.cpp")
target_include_directories(dram_simlet PUBLIC ${LEGOSIM_ROOT}/interchiplet/includes)
target_link_libraries(dram_simlet ${LEGOSIM_ROOT}/interchiplet/lib/libinterchiplet_c.a pthread)
