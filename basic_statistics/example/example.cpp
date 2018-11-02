#include "../include/basic_statistics.h"
#include "timer.h"
#include <iostream>

#include <math.h>
#include <assert.h>
#include <random>
#include <vector>
#include <cuda_profiler_api.h>



using namespace std;
using basic_statistics::Basic_Statistics;
using basic_statistics::UInt;
using basic_statistics::precision;
using timer::now;


int main()
{
    if (sizeof(precision) == 8)
    {
        cout << "use 64-bit precision." << endl;
    }
    if (sizeof(precision) == 4)
    {
        cout << "use 32-bit precision." << endl;
    }

    std::mt19937 rng(1000);
    std::normal_distribution<double> normal_dist;

    UInt n_row = 7000;
    UInt n_col = 1000;

    std::vector<precision> M(n_row * n_col);
    for (auto & x : M)
    {
        x = normal_dist(rng);
    }

    std::vector<precision> colmean(n_col);
    std::vector<precision> colvar(n_col);
    //std::vector<precision> cov(n_col * 4 / 2);

    Basic_Statistics stat(M.data(), n_row, n_col);
    stat.mean(colmean.data());
    stat.var(colvar.data());
    //stat.cov_within_group(cov.data(), 2);


    std::cout << "check point (mean) : " << colmean[0] << " " << colmean[n_col - 1] << std::endl;
    std::cout << "check point (var) : " << colvar[0] << " " << colvar[n_col - 1] << std::endl;

    // test multiplier performance
    size_t N = 2000;
    // generate multiplier
    std::vector<std::vector<precision>> multiplier(N);
    std::cout << "simulating multiplier ... " ;
    for (UInt n = 0; n < N; ++n)
    {
        multiplier[n] = std::vector<precision>(n_row);
        for (UInt i = 0; i < n_row; ++i)
        {
            multiplier[n][i] = normal_dist(rng);
        }
    }
    std::cout << " done." << std::endl;

    // doing benchmark
    cudaProfilerStart();
    auto now = timer::now();
    for (UInt n = 0; n < N; ++n)
    {
        stat.apply_multiplier(multiplier[n]);
        stat.mean(colmean.data());
        stat.var(colvar.data());
    }
    std::cout << "GPU elapsed time: " << timer::time_elapsed(now) << "seconds " << std::endl;
    cudaProfilerStop();
    std::cout << "after applying multiplier " << std::endl;
    std::cout << "check point (mean) : " << colmean[0] << " " << colmean[n_col - 1] << std::endl;
    std::cout << "check point (var) : " << colvar[0] << " " << colvar[n_col - 1] << std::endl;
    std::cout << " n_Booststrap " << N << std::endl;








    return 0;
}
