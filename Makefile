# PATH=/opt/cuda_all/cuda_11.8
PATH_TO_CUDA=/opt/cuda
INCLUDE=$(PATH_TO_CUDA)/include
LIB=$(PATH_TO_CUDA)/lib64

rel_unified:
	$(PATH_TO_CUDA)/bin/nvcc -std=c++17 -arch=sm_70 -O2 -I$(INCLUDE) compare_fft_unfied_mem.cu -o compare_fft_unfied_mem.bin -L$(LIB) -lfftw3 -lcufft 
rel_device:
	$(PATH_TO_CUDA)/bin/nvcc -std=c++17 -arch=sm_70 -O2 -I$(INCLUDE) compare_fft.cu -o compare_fft.bin -L$(LIB) -lfftw3 -lcufft	
rel_ptr:
	$(PATH_TO_CUDA)/bin/nvcc -std=c++17 -arch=sm_70 -O2 -I$(INCLUDE) compare_fft_unfied_mem_ptr.cu -o compare_fft_unfied_mem_ptr.bin -L$(LIB) -lfftw3 -lcufft	
