#include <fftw3.h>
#include <complex>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cufft.h>
#include "cuda_safe_call.h"
#include "cufft_safe_call.h"
#include <chrono>

template<typename T>
T norm(const std::vector<T>& data)
{
    T sum{0};
    for(const auto &x: data)
    {
        sum += x*x;
    }

    return std::sqrt(sum);
}

template<typename T>
T norm(const std::vector< std::complex<T> >& data)
{
    T sum{0};
    for(const auto &x: data)
    {
        sum += real(x*conj(x));
    }

    return std::sqrt(sum);
}

template <class Vec>
void plot_vec(const Vec& data)
{
    for(auto x: data)
    {
        std::cout << x << " ";
    }
    std::cout << std::endl;
}

template <class VecDev, class VecHost>
void device_to_std_vec(const VecDev& vec_d, VecHost& vec_h)
{
    using T = typename VecHost::value_type;
    std::size_t n_el = vec_h.size();
    std::size_t count = n_el*sizeof(T);
    auto host_ref = vec_h.data();
    auto dev_ref = thrust::raw_pointer_cast( vec_d.data() );
    CUDA_SAFE_CALL( cudaMemcpy( (void*) host_ref, (const void*) dev_ref, count, cudaMemcpyDeviceToHost ) );
    
}


template <class VecDev>
std::vector< typename VecDev::value_type > device_to_std_vec(const VecDev& vec_d)
{
    using T = typename VecDev::value_type;
    std::size_t n_el = vec_d.size();
    std::size_t count = n_el*sizeof(T);
    std::vector< T > vec_h(n_el);
    auto host_ref = vec_h.data();
    auto dev_ref = thrust::raw_pointer_cast( vec_d.data() );
    CUDA_SAFE_CALL( cudaMemcpy( (void*) host_ref, (const void*) dev_ref, count, cudaMemcpyDeviceToHost ) );
    return vec_h; 
}

template<typename T>
struct select_cufft_type
{
};
template<>
struct select_cufft_type<float>
{
    using real = cufftReal;
    using complex = cufftComplex;
};
template<>
struct select_cufft_type<double>
{
    using real = cufftDoubleReal;
    using complex = cufftDoubleComplex;
};



int main(int argc, char const *argv[])
{
    using T = double;
    using TRealCufft = select_cufft_type<T>::real;
    using TComplexCufft = select_cufft_type<T>::complex;

    if(argc != 4)
    {
        std::cout << "usage: " << argv[0] << " N CPU GPU_number" << std::endl;
        std::cout << "  where N is the size of a cube," << std::endl;
        std::cout << "  CPU=y/n is use CPU fftw for verification or not." << std::endl;
        return 0;
    }
    int device_id = std::atoi(argv[3]);
    cudaDeviceProp device_prop;
    CUDA_SAFE_CALL( cudaGetDeviceProperties(&device_prop, device_id) ); 	
    std::cout << "using CUDA device number " << device_id << ": " << device_prop.name << std::endl;
    CUDA_SAFE_CALL( cudaSetDevice(device_id) );

    std::size_t N_size = std::atoi(argv[1]);
    char use_fftw = argv[2][0];
    std::size_t N = N_size, M = N_size, L = N_size;
    std::size_t L_reduced = L/2+1; 

    std::vector< std::complex<T> > data_c(N*M*L_reduced);
    std::vector< T > data_r_1(N*M*L);
    std::vector< T > data_r_2(N*M*L);

    std::random_device rd;
    std::mt19937 engine{ rd() }; 
    std::uniform_real_distribution<> dist(0.0, 100.0);

    auto gen_rand = [&dist, &engine]()
    {
        return dist(engine);
    };
    std::generate(begin(data_r_1), end(data_r_1), gen_rand);
    
    //timers
 
    //FFTW part

    
    std::cout << "executing fftw...";
    std::cout << std::flush;

    fftw_complex* c_fftw = (fftw_complex*)( data_c.data() );
    T* r_fftw = data_r_1.data();
    T* r2_fftw = data_r_2.data();

    fftw_plan plan_fftw_r2c, plan_fftw_c2r;
    auto start_0 = std::chrono::high_resolution_clock::now();
    if(use_fftw == 'y')
    {
        plan_fftw_r2c = fftw_plan_dft_r2c_3d(N, M, L, r_fftw, c_fftw, FFTW_ESTIMATE);
        plan_fftw_c2r = fftw_plan_dft_c2r_3d(N, M, L, c_fftw, r2_fftw, FFTW_ESTIMATE);
    
        fftw_execute(plan_fftw_r2c);
        fftw_execute(plan_fftw_c2r);
        fftw_destroy_plan(plan_fftw_r2c);
        fftw_destroy_plan(plan_fftw_c2r);
    }
    auto stop_0 = std::chrono::high_resolution_clock::now();


    if(use_fftw == 'y')
    {
        std::transform(data_r_2.cbegin(), data_r_2.cend(), data_r_2.begin(), [&N, &M, &L]( T c) { return c/(N*M*L); });
    }

    
    std::cout << "done." << std::endl;
    //CUDA part
    std::cout << "executing cufft...";
    std::cout << std::flush;

    TRealCufft *data_r_1_dev, *data_r_2_dev;
    TComplexCufft* data_c_dev;
    std::size_t size_real = N*M*L;
    std::size_t size_complex = N*M*L_reduced;

    CUDA_SAFE_CALL( cudaMallocManaged((void**)&data_r_1_dev, sizeof(TRealCufft)*size_real ) );
    CUDA_SAFE_CALL( cudaMallocManaged((void**)&data_r_2_dev, sizeof(TRealCufft)*size_real ) );
    CUDA_SAFE_CALL( cudaMallocManaged((void**)&data_c_dev, sizeof(TComplexCufft)*size_complex ) );

    CUDA_SAFE_CALL( cudaMemcpy ( data_r_1_dev, data_r_1.data(), sizeof(TRealCufft)*size_real, cudaMemcpyHostToDevice ) );
    CUDA_SAFE_CALL( cudaMemcpy ( data_r_2_dev, data_r_2.data(), sizeof(TRealCufft)*size_real, cudaMemcpyHostToDevice ) );


    auto data_c_dev_c = data_c_dev;
    auto data_r_1_dev_c = data_r_1_dev;
    auto data_r_2_dev_c = data_r_2_dev;
    

    cufftHandle cufft_handle_r2c, cufft_handle_c2r;

    cudaEvent_t start_1, stop_1;
    CUDA_SAFE_CALL( cudaEventCreate(&start_1) );
    CUDA_SAFE_CALL( cudaEventCreate(&stop_1) );    

    CUDA_SAFE_CALL( cudaEventRecord(start_1) );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    CUFFT_SAFE_CALL( cufftPlan3d(&cufft_handle_r2c, N, M, L, CUFFT_D2Z) );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    CUFFT_SAFE_CALL( cufftPlan3d(&cufft_handle_c2r, N, M, L, CUFFT_Z2D) ); 
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    CUFFT_SAFE_CALL( cufftExecD2Z(cufft_handle_r2c, data_r_1_dev_c, data_c_dev_c ) ); //only works with C-style cast!
    CUFFT_SAFE_CALL( cufftExecZ2D(cufft_handle_c2r, data_c_dev_c, data_r_2_dev_c ) ); //only works with C-style cast!
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    CUFFT_SAFE_CALL( cufftDestroy(cufft_handle_r2c) );
    CUFFT_SAFE_CALL( cufftDestroy(cufft_handle_c2r) );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    CUDA_SAFE_CALL( cudaEventRecord(stop_1) );
    CUDA_SAFE_CALL( cudaEventSynchronize(stop_1) );
    
    std::cout << "done." << std::endl;

    float duration_1 = 0;
    CUDA_SAFE_CALL( cudaEventElapsedTime(&duration_1, start_1, stop_1) );
    auto duration_0 = std::chrono::duration_cast<std::chrono::milliseconds>(stop_0 - start_0);
    std::cout << " fftw time = " << duration_0.count() << " cufft time = " << duration_1 << std::endl;
    



    cudaFree(data_r_1_dev);
    cudaFree(data_r_2_dev);
    cudaFree(data_c_dev);



    return 0;

}


