cmake_minimum_required(VERSION 3.9)
set(project_name fft_tests)
project(${project_name} LANGUAGES CUDA CXX)
set(CMAKE_CUDA_ARCHITECTURES 35 52 61 70 75 80)
set(CMAKE_CONFIGURATION_TYPES Debug Release)
enable_language(CUDA)


# include_directories(${project_name} ${CMAKE_CURRENT_SOURCE_DIR}/../ )

add_executable(compare_fft.bin compare_fft.cu)
target_link_libraries(compare_fft.bin cufft fftw3 )

add_executable(compare_fft_unfied_mem.bin compare_fft_unfied_mem.cu)
target_link_libraries(compare_fft_unfied_mem.bin cufft fftw3 )

add_executable(compare_fft_unfied_mem_ptr.bin compare_fft_unfied_mem_ptr.cu)
target_link_libraries(compare_fft_unfied_mem_ptr.bin cufft fftw3 )
