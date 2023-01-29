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
        std::transform(data_r_2.cbegin(), data_r_2.cend(), data_r_2.begin(), [&N, &M, &L]( T c) { return c/(N*M*L); });

    
    std::cout << "done." << std::endl;
    //CUDA part
    std::cout << "executing cufft...";
    std::cout << std::flush;

    thrust::universal_vector< T > data_r_1_dev(data_r_1); //input vector copy to device

    thrust::universal_vector< thrust::complex<T> > data_c_dev(N*M*L_reduced);
    thrust::universal_vector< T > data_r_2_dev(data_r_2);

    auto data_c_dev_c = thrust::raw_pointer_cast( data_c_dev.data() );
    auto data_r_1_dev_c = thrust::raw_pointer_cast( data_r_1_dev.data() );
    auto data_r_2_dev_c = thrust::raw_pointer_cast( data_r_2_dev.data() );
    

    cufftHandle cufft_handle_r2c, cufft_handle_c2r;

    cudaEvent_t start_1, stop_1;
    CUDA_SAFE_CALL(cudaEventCreate(&start_1));
    CUDA_SAFE_CALL(cudaEventCreate(&stop_1));    

    CUDA_SAFE_CALL(cudaEventRecord(start_1));
// typedef enum cufftType_t {
//     CUFFT_R2C = 0x2a,  // Real to complex (interleaved) 
//     CUFFT_C2R = 0x2c,  // Complex (interleaved) to real 
//     CUFFT_C2C = 0x29,  // Complex to complex (interleaved) 
//     CUFFT_D2Z = 0x6a,  // T to T-complex (interleaved) 
//     CUFFT_Z2D = 0x6c,  // T-complex (interleaved) to T 
//     CUFFT_Z2Z = 0x69   // T-complex to T-complex (interleaved)
// } cufftType;
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUFFT_SAFE_CALL( cufftPlan3d(&cufft_handle_r2c, N, M, L, CUFFT_D2Z) );
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUFFT_SAFE_CALL( cufftPlan3d(&cufft_handle_c2r, N, M, L, CUFFT_Z2D) ); 
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    CUFFT_SAFE_CALL( cufftExecD2Z(cufft_handle_r2c, static_cast<TRealCufft*>( data_r_1_dev_c ), (TComplexCufft*)( data_c_dev_c ) ) ); //only works with C-style cast!
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUFFT_SAFE_CALL( cufftExecZ2D(cufft_handle_c2r, (TComplexCufft*)( data_c_dev_c ), static_cast<TRealCufft*>( data_r_2_dev_c ) ) ); //only works with C-style cast!
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    CUFFT_SAFE_CALL( cufftDestroy(cufft_handle_r2c) );
    CUFFT_SAFE_CALL( cufftDestroy(cufft_handle_c2r) );
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    CUDA_SAFE_CALL(cudaEventRecord(stop_1));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop_1));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    std::cout << "done." << std::endl;

    //check diff
    thrust::host_vector< thrust::complex<T> > data_c_host(data_c_dev);

    float duration_1 = 0;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&duration_1, start_1, stop_1));
    auto duration_0 = std::chrono::duration_cast<std::chrono::milliseconds>(stop_0 - start_0);
    std::cout << " fftw time = " << duration_0.count() << " cufft time = " << duration_1 << std::endl;
    


    //thrust::host_vector< T > data_r_2_host(data_r_2_dev);
    
    auto data_r_2_host = device_to_std_vec(data_r_2_dev);

    std::vector<T> data_r_2_from_cuda(N*M*L);

    device_to_std_vec(data_r_2_dev, data_r_2_from_cuda);


    //plot_vec(data_c);
    //std::cout << "..." << std::endl;
    //plot_vec(data_c_host);


    std::transform(data_r_2_host.cbegin(), data_r_2_host.cend(), data_r_2_host.begin(), [&N, &M, &L]( T c) { return c/(N*M*L); });  


    std::vector< std::complex<T> > diff_c_cufft_vs_fftw(N*M*L_reduced);
    std::transform(data_c.begin(), data_c.end(), data_c_host.begin(), diff_c_cufft_vs_fftw.begin(), [](auto c, auto d){ return static_cast<std::complex<T> >(c) - static_cast<std::complex<T> >(d); } );

    auto c_diff = norm(diff_c_cufft_vs_fftw);  
    std::cout << "cufft vs fftw complex difference: " << c_diff << std::endl;
    if (c_diff/(N*M*L)>1.0e-10)
    {
        std::cout << "fftw(u(0)) = " << data_c[0] << " cufft(u(0)) = " << data_c_host[0] << std::endl;
    }



    std::vector<T> diff_r_fftw(N*M*L);
    std::transform(data_r_1.begin(), data_r_1.end(), data_r_2.begin(), diff_r_fftw.begin(), std::minus< T >() );
    std::vector<T> diff_r_cufft(N*M*L);
    std::transform(data_r_1.begin(), data_r_1.end(), data_r_2_host.begin(), diff_r_cufft.begin(), std::minus< T >() );

    std::vector<T> diff_r_fftw_vs_cufft(N*M*L);
     std::transform(data_r_2.begin(), data_r_2.end(), data_r_2_host.begin(), diff_r_fftw_vs_cufft.begin(), std::minus< T >() );   
    

    std::cout << "fftw complex vector norm: " << norm(data_c) << std::endl;

    std::cout << "fftw result difference: " << norm(diff_r_fftw) << std::endl;
    std::cout << "cufft result difference: " << norm(diff_r_cufft) << std::endl;
    std::cout << "cufft vs fftw result difference: " << norm(diff_r_fftw_vs_cufft) << std::endl;



    return 0;

}


