cmake_minimum_required (VERSION 2.6)


project(RLCU)


find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS})

aux_source_directory(. DIR_SRCS)

add_executable(RLCU ${DIR_SRCS})

target_link_libraries(RLCU clFFT clRNG ${OPENCL_LIBRARIES})
