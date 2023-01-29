rel_unified:
	/opt/cuda/bin/nvcc -arch=sm_70 -O2 compare_fft_unfied_mem.cu -o compare_fft_unfied_mem.bin -lfftw3 -lcufft	
rel_device:
	/opt/cuda/bin/nvcc -arch=sm_70 -O2 compare_fft.cu -o compare_fft.bin -lfftw3 -lcufft	
rel_ptr:
	/opt/cuda/bin/nvcc -arch=sm_70 -O2 compare_fft_unfied_mem_ptr.cu -o compare_fft_unfied_mem_ptr.bin -lfftw3 -lcufft	
