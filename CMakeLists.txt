cmake_minimum_required(VERSION 3.24)  
project(filtered LANGUAGES C CXX)  
set(CMAKE_CXX_STANDARD 20)            

set(SOURCE_FILES src/main.cpp src/App.cpp src/acquire_audio_stream.cpp)

option(ONNX_USE_QNN "Will fetch a version of Onnx Runtime that only supports QNN and CPU" OFF)

include(cmake/liboboe.cmake)
include(cmake/DynamicONNXRuntime.cmake)
include(cmake/cxxopts.cmake)
include(cmake/magic_enum.cmake)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE ${CMAKE_SOURCE_DIR}/../oboe/include
    PRIVATE ${CMAKE_SOURCE_DIR}/../oboe_boilerplates
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE oboe
    PRIVATE log
    PRIVATE Ort::Ort
    PRIVATE cxxopts::cxxopts
    PRIVATE magic_enum::magic_enum
)

target_link_options(${PROJECT_NAME} PRIVATE -Wl,--allow-shlib-undefined)