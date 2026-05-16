cmake_minimum_required(VERSION 3.10)

project(sstStonne LANGUAGES CXX)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)

file(GLOB SOURCE_FILES ${SRC_DIR}/*.cpp)
file(GLOB HEADER_FILES ${INCLUDE_DIR}/*.h)

set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
file(MAKE_DIRECTORY ${LIB_DIR})

add_library(sstStonne STATIC ${SOURCE_FILES})
add_library(sstStonne_shared SHARED ${SOURCE_FILES})

target_include_directories(sstStonne PUBLIC ${INCLUDE_DIR} ${EXTERNAL_DIR})
target_include_directories(sstStonne_shared PUBLIC ${INCLUDE_DIR} ${EXTERNAL_DIR})

set_target_properties(sstStonne PROPERTIES OUTPUT_NAME "sstStonne")
set_target_properties(sstStonne_shared PROPERTIES OUTPUT_NAME "sstStonne")

install(TARGETS sstStonne sstStonne_shared
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY ${INCLUDE_DIR}/ DESTINATION include/sstStonne FILES_MATCHING PATTERN "*.h")

message(STATUS "sstStonne library will be built in ${LIB_DIR}")