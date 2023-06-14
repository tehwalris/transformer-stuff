cmake_minimum_required(VERSION 3.10)

project(cpp_stuff LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mf16c")

add_library(ggml OBJECT ggml.c ggml.h)

add_library(llama OBJECT llama.cpp llama.h llama-util.h)

add_library(cpp_stuff_base loading.cpp)
target_link_libraries(cpp_stuff_base PRIVATE llama ggml)

add_library(cpp_stuff_cuda cuda.cu)
set_target_properties(cpp_stuff_cuda PROPERTIES CUDA_ARCHITECTURES "80;86;87")

install(TARGETS cpp_stuff_base)
install(TARGETS cpp_stuff_cuda)