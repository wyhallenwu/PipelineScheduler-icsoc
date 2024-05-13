cmake_minimum_required(VERSION 3.18)
project(PlayGround)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Only do these if this is the main project, and not if it is included through
# add_subdirectory
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_STANDARD 17)
    # Set C++ version and optimization level
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -DNDEBUG -Wno-deprecated-declarations -O0") #-Ofast

    # Let's ensure -std=c++xx instead of -std=g++xx
    set(CMAKE_CXX_EXTENSIONS OFF)

    # Let's nicely support folders in IDE's
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    # Testing only available if this is the main app. Note this needs to be done
    # in the main CMakeLists since it calls enable_testing, which must be in the
    # main CMakeLists.
    include(CTest)

    # Docs only available if this is the main app
    find_package(Doxygen)
    # For finding FindTensorRT.cmake
    set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
    if (Doxygen_FOUND)
        add_subdirectory(docs)
    else ()
        message(STATUS "Doxygen not found, not building docs")
    endif ()
endif ()

# Specify the path to TensorRT root dir
if (NOT TensorRT_DIR)
    set(TensorRT_DIR /TensorRT/)
endif ()
# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# We require CUDA, OpenCV, and TensorRT
#find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)


# Find gRPC installation
# Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(_GRPC_GRPCPP gRPC::grpc++)
if (CMAKE_CROSSCOMPILING)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else ()
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
endif ()

add_executable(gpu_upload_chunk_vs_indiv gpu_upload_chunk_vs_indiv.cpp)
target_include_directories(gpu_upload_chunk_vs_indiv PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS})
target_link_libraries(gpu_upload_chunk_vs_indiv PUBLIC
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${TensorRT_LIBRARIES}
)