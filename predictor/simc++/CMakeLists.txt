cmake_minimum_required(VERSION 3.10)
project(LightGBMPrediction)

# Set compiler flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set LightGBM include directories
set(LIGHTGBM_INCLUDE_DIR "/usr/include/LightGBM")
include_directories(${LIGHTGBM_INCLUDE_DIR})

# Find LightGBM library
find_library(LIGHTGBM_LIBRARY
    NAMES lib_lightgbm.so
    PATHS "/usr/lib"
    NO_DEFAULT_PATH)

if(NOT LIGHTGBM_LIBRARY)
    message(FATAL_ERROR "LightGBM library not found.")
endif()

# Add the executable
add_executable(predict predict.cpp)

# Link against LightGBM library and external libraries
target_link_libraries(predict PRIVATE ${LIGHTGBM_LIBRARY})