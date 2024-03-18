cmake_minimum_required(VERSION 3.18)

project(protobufprotocols)

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

# Proto file
get_filename_component(data_proto "pipelinescheduler.proto" ABSOLUTE)
get_filename_component(data_proto_path "${data_proto}" PATH)

# Generated sources
set(data_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/pipelinescheduler.pb.cc")
set(data_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/pipelinescheduler.pb.h")
set(data_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/pipelinescheduler.grpc.pb.cc")
set(data_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/pipelinescheduler.grpc.pb.h")
add_custom_command(
        OUTPUT "${data_proto_srcs}" "${data_proto_hdrs}" "${data_grpc_srcs}" "${data_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${data_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${data_proto}"
        DEPENDS "${data_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# data_grpc_proto
add_library(data_grpc_proto
        ${data_grpc_srcs}
        ${data_grpc_hdrs}
        ${data_proto_srcs}
        ${data_proto_hdrs})
target_link_libraries(data_grpc_proto
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF})

# Proto file
get_filename_component(device_proto "indevicecommunication.proto" ABSOLUTE)
get_filename_component(device_proto_path "${device_proto}" PATH)

# Generated sources
set(device_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/indevicecommunication.pb.cc")
set(device_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/indevicecommunication.pb.h")
set(device_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/indevicecommunication.grpc.pb.cc")
set(device_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/indevicecommunication.grpc.pb.h")
add_custom_command(
        OUTPUT "${device_proto_srcs}" "${device_proto_hdrs}" "${device_grpc_srcs}" "${device_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${data_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${device_proto}"
        DEPENDS "${device_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# device_grpc_proto
add_library(device_grpc_proto
        ${device_grpc_srcs}
        ${device_grpc_hdrs}
        ${device_proto_srcs}
        ${device_proto_hdrs})
target_link_libraries(device_grpc_proto
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF})


set(
        HEADER_FILES
        data_grpc_hdrs
        device_grpc_hdrs
)


# Here we configure variables specifying the target directories.
set(INSTALL_RUNTIME_DIR ${CMAKE_INSTALL_BINDIR})
set(INSTALL_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/cmake)
set(INSTALL_LIBRARY_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE})
set(INSTALL_ARCHIVE_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE}/static)
set(INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_PREFIX}/protobufprotocols)

# Define the configuration file locations.
set(PROJECT_CONFIG_VERSION_FILE "${PROJECT_BINARY_DIR}/protobufprotocols-config-version.cmake")
set(PROJECT_CONFIG_FILE "${PROJECT_BINARY_DIR}/protobufprotocols-config.cmake")


configure_package_config_file(cmake/protobufprotocols-config.cmake.in
        ${PROJECT_CONFIG_FILE}
        INSTALL_DESTINATION ${INSTALL_CONFIG_DIR})


# The INSTALL(TARGET) command specifies where to install specific targets.
# By specifying EXPORT here, we add these targets to an export group.
install(TARGETS data_grpc_proto device_grpc_proto
        EXPORT protobufprotocols-targets
        RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
        ARCHIVE DESTINATION ${INSTALL_ARCHIVE_DIR})

# The INSTALL(FILES) command specifies where to install specific files, such as headers.
install(FILES ${HEADER_FILES}
        DESTINATION ${INSTALL_INCLUDE_DIR})

# The INSTALL(FILES) command specifies where to install specific files, such as headers.
install(FILES
        ${PROJECT_CONFIG_VERSION_FILE}
        ${PROJECT_CONFIG_FILE}
        DESTINATION ${INSTALL_CONFIG_DIR})

# The INSTALL(EXPORT) command installs a previously defined export group to a script.
install(EXPORT protobufprotocols-targets
        FILE protobufprotocols-targets.cmake
        NAMESPACE protobufprotocols::
        DESTINATION ${INSTALL_CONFIG_DIR})

export(EXPORT protobufprotocols-targets
        FILE ${CMAKE_CURRENT_BINARY_DIR}/protobufprotocols-targets.cmake
        NAMESPACE protobufprotocols::)

# Register the package in the user's package registry.
# https://cmake.org/cmake/help/v3.11/command/export.html
export(PACKAGE protobufprotocols)