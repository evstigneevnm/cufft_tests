cmake_minimum_required(VERSION 3.16)
set(project_name fft_tests)
project(${project_name} LANGUAGES CUDA CXX)
set(CMAKE_CUDA_ARCHITECTURES 35 52 61 70 75 80)
set(CMAKE_CONFIGURATION_TYPES Debug Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)

# include_directories(${project_name} ${CMAKE_CURRENT_SOURCE_DIR}/../ )

add_executable(compare_fft_unfied_mem_manage_buffers.bin compare_fft_unfied_mem_manage_buffers.cu)
target_compile_options(compare_fft_unfied_mem_manage_buffers.bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
target_link_libraries(compare_fft_unfied_mem_manage_buffers.bin PRIVATE TBB::tbb OpenMP::OpenMP_CXX cufft fftw3 )

add_executable(compare_fft.bin compare_fft.cu)
# target_compile_options(compare_fft.bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
target_link_libraries(compare_fft.bin PRIVATE PRIVATE TBB::tbb cufft fftw3 )

add_executable(compare_fft_unfied_mem.bin compare_fft_unfied_mem.cu)
# target_compile_options(compare_fft_unfied_mem.bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
target_link_libraries(compare_fft_unfied_mem.bin PRIVATE PRIVATE TBB::tbb cufft fftw3 )

add_executable(compare_fft_unfied_mem_ptr.bin compare_fft_unfied_mem_ptr.cu)
# target_compile_options(compare_fft_unfied_mem_ptr.bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
# target_link_libraries(compare_fft_unfied_mem_ptr.bin PRIVATE OpenMP::OpenMP_CXX cufft fftw3 )
target_link_libraries(compare_fft_unfied_mem_ptr.bin PRIVATE PRIVATE TBB::tbb cufft fftw3 )

