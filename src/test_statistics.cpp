#include "conditional_moment_ineq.h"
#include "eigen_notation.h"
#include "basic_statistics.h"
#include "quadprog.h"

using basic_statistics::Basic_Statistics;
using namespace conditional_moment_inequality;

constexpr FLOAT Conditional_Moment_Ineq::inf;

inline FLOAT S1(const FLOAT * m, const FLOAT * sigma_var, const vector<int> & sign)
{
    UInt dim_M = sign.size();
    FLOAT s = 0;
    #pragma omp simd reduction(+:s)
    for (UInt j = 0; j < dim_M; ++j)
    {
        if (sign[j] == 0)
        {
            s += m[j] * m[j] / sigma_var[j];
        }
        if (sign[j] == 1)
        {
            s += (m[j] < 0) ? m[j] * m[j] / sigma_var[j] : 0;
        }
        if (sign[j] == -1)
        {
            s += (m[j] > 0) ? m[j] * m[j] / sigma_var[j] : 0;
        }
    }
    return s;
}

inline FLOAT S2(const FLOAT * m, const FLOAT * sigma_cov, const vector<int> & sign)
{
    UInt dim_M = sign.size();
    ConstMapMat Sigma(sigma_cov, dim_M, dim_M);
    ConstMapVec M(m, dim_M);

    Mat G = 2 * Sigma.inverse();
    Vec c = -2 * Sigma.llt().solve(M);

    if (G.hasNaN()) throw std::runtime_error("NaN value detected.");
    if (c.hasNaN()) throw std::runtime_error("NaN value detected.");

    FLOAT optf = QuadProg::quadprog(G, c, sign);
    optf += M.transpose() * Sigma.llt().solve(M);

    return optf;
}


inline FLOAT S_criteria(const FLOAT * m, const FLOAT * sigma_var, const FLOAT * sigma_cov, const vector<int> & sign, int stat_type)
{
    FLOAT value = 0;
    if (stat_type == Conditional_Moment_Ineq::CvM_S1 or stat_type == Conditional_Moment_Ineq::KS_S1)
    {
        value = S1(m, sigma_var, sign);
    }

    if (stat_type == Conditional_Moment_Ineq::CvM_S2 or stat_type == Conditional_Moment_Ineq::KS_S2)
    {
        value = S2(m, sigma_cov, sign);
    }
    return value;
}

FLOAT Conditional_Moment_Ineq::S(const FLOAT *M, const FLOAT *Sigma_var, const FLOAT * Sigma_cov) const
{
    check(0 <= stat_type and stat_type <= KS_S2, "type of statistics should be valid");

    UInt dim_ins = instruments[0].size();
    vector<FLOAT> S_of_all_instruments(dim_ins);

    if (stat_type == CvM_S1 or stat_type == KS_S1)
    {
        assert(Sigma_var != nullptr);

#pragma omp parallel for schedule(dynamic)
        for (UInt i = 0; i < dim_ins; ++i)
        {
            const FLOAT * m = M + i * dim_M;
            const FLOAT * sigma = Sigma_var + i * dim_M;
            S_of_all_instruments[i] = S_criteria(m, sigma, nullptr, sign, stat_type);
        }
    }

    if (stat_type == CvM_S2 or stat_type == KS_S2)
    {
        assert(Sigma_cov != nullptr);

#pragma omp parallel for schedule(dynamic)
        for (UInt i = 0; i < dim_ins; ++i)
        {
            const FLOAT * m = M + i * dim_M;
            const FLOAT * sigma = Sigma_cov + i * dim_M * dim_M;
            S_of_all_instruments[i] = S_criteria(m, nullptr, sigma, sign, stat_type);
        }
    }

    FLOAT s = 0;
    if (stat_type == CvM_S1 or stat_type == CvM_S2)
    {
        check(weight.size() == dim_ins, "dimension of weight should match");
#pragma omp simd reduction(+:s)
        for (UInt i = 0; i < dim_ins; ++i) s += S_of_all_instruments[i] * weight[i];
    }

    if (stat_type == KS_S1 or stat_type == KS_S2)
    {
        s = *std::max_element(S_of_all_instruments.begin(), S_of_all_instruments.end());
    }
    return s;
}

FLOAT Conditional_Moment_Ineq::test_statistics() const
{
    update_cross_product_cache();
    prepare_GMS();
    UInt n_instruments = instruments[0].size();
    Mat sigma_var = ConstMapMat(var_M_instrument_prod, dim_M, n_instruments);
    sigma_var.colwise() += epsilon * ConstMapVec(var_M, dim_M);

    Mat m = sqrt_sample_size * ConstMapMat(mean_M_instrument_prod, dim_M, n_instruments);

    FLOAT test_stat;

    if (stat_type == CvM_S1 or stat_type == KS_S1)
    {
        test_stat = S(m.data(), sigma_var.data(), nullptr);
    }

    if (stat_type == CvM_S2 or stat_type == KS_S2)
    {
        Basic_Statistics stat(M_instrument_prod, sample_size, dim_M * n_instruments);
        vector<FLOAT> sigma_cov(n_instruments * dim_M  * dim_M);
        stat.cov_within_group(sigma_cov.data(), dim_M);
        for (UInt i = 0; i < n_instruments; ++i)
        {
            const FLOAT * sigma_var_ptr = sigma_var.data() + i * dim_M;
            for (UInt j = 0; j < dim_M; ++j)
            {
                sigma_cov[j + dim_M * j + dim_M * dim_M * i] = sigma_var_ptr[j];
            }
        }
        test_stat = S(m.data(), nullptr, sigma_cov.data());
    }

    return test_stat;
}

vector<FLOAT> Conditional_Moment_Ineq::boostrap_test_statistics(FLOAT x) const
{
    update_cross_product_cache();
    prepare_GMS();

    UInt n_instruments = instruments[0].size();
    vector<FLOAT> bootstrap_test_stat(n_bootstrap, inf);
    Basic_Statistics bootstrap_stat(M_instrument_prod, sample_size, dim_M * n_instruments);

    UInt count_greater_x = 0;
    for (UInt b = 0; b < n_bootstrap; ++b)
    {
        // get bootstrap multiplier
        const FLOAT * multiplier = bootstrap_multiplier[b];

        // mean and variance of cross product of M and instrument with boostraped sample
        Mat mean_boostrap_stat(dim_M, n_instruments);
        Mat var_boostrap_stat(dim_M, n_instruments);

        bootstrap_stat.apply_multiplier(multiplier);
        bootstrap_stat.var(var_boostrap_stat);
        bootstrap_stat.mean(mean_boostrap_stat);

        // Now start to calculate the test statistics
        // epsilon reguliarizaiton
        var_boostrap_stat.colwise() += epsilon * ConstMapVec(var_M, dim_M);

        // adjust variance of bootstrapped sample, ( when calculting the covariance matrix, the mean of the original sample is used)
        // the result should be valid with or without the following when the sample size goes to infinity.
        var_boostrap_stat.array() += (mean_boostrap_stat - ConstMapMat(mean_M_instrument_prod, dim_M, n_instruments)).array().square();

        // adjust mean with moment selection function
        mean_boostrap_stat -= ConstMapMat(mean_M_instrument_prod, dim_M, n_instruments);
        mean_boostrap_stat *= sqrt_sample_size;
        mean_boostrap_stat += ConstMapMat(GMS, dim_M, n_instruments);

        if (stat_type == CvM_S1 or stat_type == KS_S1)
        {
            bootstrap_test_stat[b] = S(mean_boostrap_stat.data(),
                                       var_boostrap_stat.data(), nullptr);
        }
        if (stat_type == CvM_S2 or stat_type == KS_S2)
        {
            vector<FLOAT> sigma_cov(n_instruments * dim_M  * dim_M);
            bootstrap_stat.cov_within_group(sigma_cov.data(), dim_M);
            for (UInt i = 0; i < n_instruments; ++i)
            {
                const FLOAT * sigma_var_ptr = var_boostrap_stat.data() + i * dim_M;
                for (UInt j = 0; j < dim_M; ++j)
                {
                    sigma_cov[j + dim_M * j + dim_M * dim_M * i] = sigma_var_ptr[j];
                }
            }
            bootstrap_test_stat[b] = S(mean_boostrap_stat.data(), nullptr, sigma_cov.data());
        }

        count_greater_x += bootstrap_test_stat[b] > x;
        // if there is enough bootstrap statistics which are greater than x,
        // we know P( bootstrap statistics <= x) < 1 - alpha for sure.
        // Hence, there is no need to calculate the rest bootstrap samples.
        // Thus, by add the following line, we avoid redundant computation when
        // the test accept the null hypothesis.
        if (count_greater_x > alpha * n_bootstrap) break;
    }

//    printf("o\n"); fflush(stdout);

    return bootstrap_test_stat;
}

inline
FLOAT empirical_cdf(const vector<FLOAT> & v, FLOAT x)
{
    UInt c = 0;
    for (UInt i = 0; i < v.size(); ++i)
    {
        c += (v[i] < x);
    }
    return (FLOAT) c / (FLOAT) v.size();
}


bool Conditional_Moment_Ineq::inference() const
{
    check(1 - alpha > 0.5, "alpha should be less than 0.5");
    check(alpha > 0, "alpha should be non-negative.");

    FLOAT test_stat = test_statistics();
    vector<FLOAT> bootstrap_distribution = boostrap_test_statistics(test_stat);

    return 1 - alpha >= empirical_cdf(bootstrap_distribution, test_stat);
}

FLOAT Conditional_Moment_Ineq::p_value() const
{
    check(1 - alpha > 0.5, "alpha should be less than 0.5");
    check(alpha > 0, "alpha should be non-negative.");

    FLOAT test_stat = test_statistics();
    vector<FLOAT> bootstrap_distribution = boostrap_test_statistics();

    return 1 - empirical_cdf(bootstrap_distribution, test_stat);
}
