#include "conditional_moment_ineq.h"

using namespace conditional_moment_inequality;
using bd_dist = std::binomial_distribution<UInt>;

// Nonparametric Bootstrap Multiplier
FLOAT*
bootstrap_multiplier_sampler(UInt sample_size, mt19937& rng)
{
    FLOAT* multiplier = new FLOAT[sample_size];

    UInt remaining_draws = sample_size;
    for (UInt i = 0; i < sample_size; ++i)
    {
        double  p = 1.0 / (sample_size - i);
        bd_dist binomial_dist(remaining_draws, p);

        UInt temp     = binomial_dist(rng);
        multiplier[i] = (FLOAT)temp;
        remaining_draws -= temp;
    }
    check(remaining_draws == 0, "nothing should left");
    return multiplier;
}

void
Conditional_Moment_Ineq::gen_bootstrap_multiplier()
{
    free_bootstrap_multiplier();
    bootstrap_multiplier.clear();
    mt19937 rng(2018);
    for (UInt i = 0; i < n_bootstrap; ++i)
    {
        bootstrap_multiplier.push_back(bootstrap_multiplier_sampler(sample_size, rng));
    }
}
