cmake_minimum_required(VERSION 3.18)
project(PipePlusPlus)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Only do these if this is the main project, and not if it is included through
# add_subdirectory
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_STANDARD 17)
    # Set C++ version and optimization level
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -Wall -DNDEBUG -Wno-deprecated-declarations -O0") #-Ofast

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
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Find Protobuf installation
# Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.
option(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_REFLECTION gRPC::grpc++_reflection)
if (CMAKE_CROSSCOMPILING)
    find_program(_PROTOBUF_PROTOC protoc)
else ()
    set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
endif ()

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

find_package(spdlog REQUIRED)

add_subdirectory(libs/misc)
add_subdirectory(libs/microservice)
add_subdirectory(libs/trtengine)
add_subdirectory(libs/onnx2trt)
add_subdirectory(libs/baseprocessors)
add_subdirectory(libs/protobufprotocols)
add_subdirectory(libs/communicator)
add_subdirectory(libs/container_agent)
add_subdirectory(libs/data_source)
add_subdirectory(libs/yolov5)
add_subdirectory(libs/retinaface)
# add_subdirectory(libs/arcface)
add_subdirectory(libs/device_agent)
add_subdirectory(libs/controller)

add_library(PipePlusPlus
        src/main.cpp)

target_include_directories(PipePlusPlus PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../libs/protobufprotocols>)
target_link_libraries(PipePlusPlus PUBLIC
        data_grpc_proto
        device_grpc_proto
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${TensorRT_LIBRARIES}
        ${_GRPC_GRPCPP}
        ${_REFLECTION}
        ${_PROTOBUF_LIBPROTOBUF}
        misc
        microservice
        communicator
        data_source
        container_agent
)
add_executable(run src/main.cpp)
target_link_libraries(run PipePlusPlus)

add_executable(DeviceAgent libs/device_agent/device_agent.cpp)
target_link_libraries(DeviceAgent device_agent ${CUDA_LIBRARIES} CUDA::nvml)

add_executable(Controller libs/controller/controller.cpp)
target_link_libraries(Controller controller)

add_executable(Container_DataSource libs/data_source/data_source.cpp)
target_link_libraries(Container_DataSource data_source)

add_executable(Container_Yolov5 libs/yolov5/yolov5.cpp)
target_link_libraries(Container_Yolov5 yolov5)

add_executable(convert_onnx2trt libs/onnx2trt/onnx2trt.cpp)
target_link_libraries(convert_onnx2trt onnx2trt)

add_executable(Container_RetinaFace libs/retinaface/retinaface.cpp)
target_link_libraries(Container_RetinaFace retinaface)

#add_executable(Container_ArcFace libs/arcface/arcface.cpp)
#target_link_libraries(Container_ArcFace arcface)