# cufft_tests
This is the side test project to verify why cufft fails for large arrays.

Found the problem - one must allocate buffers for cufft manually not in GPU device memory (not enough for arrays of sizes 1000^3 and more), but e.g. in mannaged (unified) memory. For more information see the code inside.
