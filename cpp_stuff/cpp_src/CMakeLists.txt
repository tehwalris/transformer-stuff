cmake_minimum_required(VERSION 3.10)

project(cpp_stuff)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mf16c")

add_library(cpp_stuff_base ggml.c llama.cpp loading.cpp)

install(TARGETS cpp_stuff_base)