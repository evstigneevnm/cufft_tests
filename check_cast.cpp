#include <iostream>
#include <chrono>
#include <vector>


int main()
{
    std::size_t sz = 10000000;
    std::int64_t value_0 = 1, value_1 = 1, value_2 = 1;
    std::vector<std::size_t> some_array_0(10000, 2);

    auto start_0 = std::chrono::high_resolution_clock::now();
   
    for(std::size_t j=0;j<sz;j++)
    {
        value_0 += static_cast<std::int64_t>(j)*static_cast<std::int64_t>(some_array_0[j%10000]);
    }

    auto stop_0 = std::chrono::high_resolution_clock::now();


    std::vector<std::size_t> some_array(10000, 2);
    
    auto start_1 = std::chrono::high_resolution_clock::now();
 
    for(std::size_t j=0;j<sz;j++)
    {
        value_1 += j*some_array[j%10000];
    }

    auto stop_1 = std::chrono::high_resolution_clock::now();

   
    std::vector<std::size_t> some_array2(10000, 2);
    
    auto start_2 = std::chrono::high_resolution_clock::now();
 
    for(std::size_t j=0;j<sz;j++)
    {
        value_2 += static_cast<std::int64_t>(j)*static_cast<std::int64_t>( some_array2[j%10000] );
    }

    auto stop_2 = std::chrono::high_resolution_clock::now();


    auto duration_0 = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_0 - start_0);
    std::cout << " warm up     " << duration_0.count() << std::endl;


    auto duration_1 = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_1 - start_1);
    std::cout << " no cast     " << duration_1.count() << std::endl;

    auto duration_2 = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_2 - start_2);
    std::cout << " static cast " << duration_2.count() << std::endl;

    std::cout << value_0 << " " << value_1 << " " << value_2 << std::endl;


    return 0;
}
