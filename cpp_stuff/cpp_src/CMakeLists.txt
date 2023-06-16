cmake_minimum_required(VERSION 3.10)

project(cpp_stuff LANGUAGES C CXX CUDA HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mf16c")

set(CMAKE_HIP_ARCHITECTURES gfx900)
find_package(rocthrust REQUIRED)
find_dependency(AMDDeviceLibs)
add_compile_options("$<$<COMPILE_LANGUAGE:HIP>:SHELL:-x hip --rocm-device-lib-path=${AMD_DEVICE_LIBS_PREFIX}/amdgcn/bitcode -std=c++17>")
set(CMAKE_INCLUDE_SYSTEM_FLAG_HIP "-isystem ")

add_library(ggml OBJECT ggml.c ggml.h)

add_library(llama OBJECT llama.cpp llama.h llama-util.h)

add_library(cpp_stuff_base loading.cpp model.cpp)
target_link_libraries(cpp_stuff_base PRIVATE llama ggml)

add_library(cpp_stuff_cuda cuda.cu)
target_compile_options(cpp_stuff_cuda PRIVATE -fvisibility=hidden)
set_target_properties(cpp_stuff_cuda PROPERTIES CUDA_ARCHITECTURES "80;86;87")
add_custom_command(TARGET cpp_stuff_cuda POST_BUILD
  COMMAND llvm-objcopy --localize-hidden $<TARGET_FILE:cpp_stuff_cuda>
)

add_library(cpp_stuff_hip hip.cpp)
target_compile_options(cpp_stuff_hip PRIVATE -fvisibility=hidden)
set_source_files_properties(hip.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(cpp_stuff_hip PRIVATE roc::rocthrust)
target_compile_definitions(cpp_stuff_hip PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
add_custom_command(TARGET cpp_stuff_hip POST_BUILD
  COMMAND llvm-objcopy --localize-hidden $<TARGET_FILE:cpp_stuff_hip>
)

install(TARGETS cpp_stuff_base)
install(TARGETS cpp_stuff_cuda)
install(TARGETS cpp_stuff_hip)