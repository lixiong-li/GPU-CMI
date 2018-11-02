#include "conditional_moment_ineq.h"
#include <set>
#include "basic_statistics.h"

using namespace conditional_moment_inequality;
using std::set;
using basic_statistics::Basic_Statistics;

constexpr int Conditional_Moment_Ineq::CvM_S1;
constexpr int Conditional_Moment_Ineq::KS_S1;
constexpr int Conditional_Moment_Ineq::CvM_S2;
constexpr int Conditional_Moment_Ineq::KS_S2;

constexpr int Conditional_Moment_Ineq::eq_zero;
constexpr int Conditional_Moment_Ineq::greater_zero;
constexpr int Conditional_Moment_Ineq::less_zero;

Conditional_Moment_Ineq::Conditional_Moment_Ineq(UInt sample_size,
                                                 FLOAT alpha,
                                                 int stat_type,
                                                 UInt n_bootstrap,
                                                 FLOAT epsilon)
  : sample_size(sample_size)
  , sqrt_sample_size(sqrt((double)sample_size))
  , alpha(alpha)
  , stat_type(stat_type)
  , n_bootstrap(n_bootstrap)
  , epsilon(epsilon)
{
    check(sample_size >= 3, "we should have at least 3 observation.");
    product_cache_need_update = true;
    own_instruments = false;

    cond_M = nullptr;
    mean_M = nullptr;
    var_M = nullptr;
    cov_M = nullptr;

    M_instrument_prod = nullptr;
    mean_M_instrument_prod = nullptr;
    var_M_instrument_prod = nullptr;

    GMS_need_update = true;
    GMS = nullptr;

    gen_bootstrap_multiplier();
}

void
Conditional_Moment_Ineq::set_cond_M(const FLOAT* cond_M, UInt dim_M, const int * sign)
{    
    this->dim_M = dim_M;
    this->cond_M = cond_M;

    if (sign == nullptr)
    {
        this->sign = vector<int>(dim_M, 1);
    }
    else
    {
        this->sign = vector<int>(sign, sign + dim_M);
    }

    update_M_cache();
    product_cache_need_update = true;
    GMS_need_update = true;
}

void
Conditional_Moment_Ineq::set_instruments(const vector<const FLOAT*>& X, UInt dim_M)
{
    free_instruments();
    instruments.clear();
    instruments.reserve(dim_M);
    auto instr = cube_instrument(X, sample_size);
    for (UInt i = 0; i < dim_M; ++i)
    {
        instruments.push_back(instr.first);
    }
    weight = instr.second;
    own_instruments = true;
    product_cache_need_update = true;
    GMS_need_update = true;
}

void
Conditional_Moment_Ineq::set_instruments(const vector<FLOAT *> &X, UInt dim_M)
{
    vector<const FLOAT * > xx;
    xx.reserve(X.size());
    for (UInt i = 0; i < X.size(); ++i)
    {
        xx.push_back(X[i]);
    }
    set_instruments(xx, dim_M);
}


void
Conditional_Moment_Ineq::set_instruments(const vector<vector<FLOAT*>>& X)
{
    vector<vector<const FLOAT*> > const_X;
    const_X.reserve(X.size());

    for (UInt i = 0; i < X.size(); ++i)
    {
        vector<const FLOAT*> const_x(X[i].size());

        for (UInt j = 0; j < X[i].size(); ++j)
        {
            const_x[j] = X[i][j];
        }
        const_X.push_back(const_x);
    }
    set_instruments(const_X);
}



void
Conditional_Moment_Ineq::set_instruments(const vector<vector<FLOAT*>>& X,
                                         const vector<FLOAT>&          weight)
{
    vector<vector<const FLOAT*> > const_X;
    const_X.reserve(X.size());

    for (UInt i = 0; i < X.size(); ++i)
    {
        vector<const FLOAT*> const_x(X[i].size());

        for (UInt j = 0; j < X[i].size(); ++j)
        {
            const_x[j] = X[i][j];
        }
        const_X.push_back(const_x);
    }
    set_instruments(const_X, weight);
}

void
Conditional_Moment_Ineq::set_instruments(const vector<vector<const FLOAT *> > &X)
{
    free_instruments();
    instruments = X;
    weight.clear();
    product_cache_need_update = true;
    GMS_need_update = true;
}

void
Conditional_Moment_Ineq::set_instruments(const vector<vector<const FLOAT *> > &X, const vector<FLOAT> &weight)
{
    free_instruments();
    instruments = X;
    this->weight = weight;
    product_cache_need_update = true;
    GMS_need_update = true;
}


Conditional_Moment_Ineq::~Conditional_Moment_Ineq()
{
    free_cross_product_cache();
    free_M_cache();
    free_instruments();
    free_GMS();
    free_bootstrap_multiplier();
}

void
Conditional_Moment_Ineq::free_instruments()
{
    if (own_instruments)
    {
        set<const FLOAT*> collection_of_instruments;
        for (auto& ins : instruments)
        {
            for (const FLOAT* ptr : ins)
            {
                collection_of_instruments.insert(ptr);
            }
        }
        for (const FLOAT* ptr : collection_of_instruments)
        {
            delete_array(ptr);
        }

        own_instruments   = false;
    }
}

void
Conditional_Moment_Ineq::free_cross_product_cache() const
{
    delete_array(M_instrument_prod);
    delete_array(mean_M_instrument_prod);
    delete_array(var_M_instrument_prod);
}

void
Conditional_Moment_Ineq::free_M_cache() const
{
    delete_array(mean_M);
    delete_array(var_M);
    delete_array(cov_M);
}

void
Conditional_Moment_Ineq::free_GMS() const
{
    delete_array(GMS);
}

void
Conditional_Moment_Ineq::free_bootstrap_multiplier()
{
    for (auto & b : bootstrap_multiplier)
    {
        delete_array(b);
    }
}

void Conditional_Moment_Ineq::update_M_cache() const
{
    free_M_cache();

    mean_M = new FLOAT[dim_M];
    var_M = new FLOAT[dim_M];
    cov_M = new FLOAT[dim_M * dim_M];

    check(cond_M != nullptr, "cond_M must be set");
    Basic_Statistics M_statistics(cond_M, sample_size, dim_M);
    M_statistics.mean(mean_M);
    M_statistics.var(var_M);
    M_statistics.cov(cov_M);

    // in case, some moments has zero variance (i.e. is degenerate)
    constexpr FLOAT small_criteria = 1e-6;
    for (UInt i = 0; i < dim_M; ++i)
    {
        if (var_M[i] < small_criteria)
        {
            var_M[i] = epsilon;
            cov_M[i + i * dim_M] = epsilon;
        }
    }
}

void Conditional_Moment_Ineq::update_cross_product_cache() const
{
    if (product_cache_need_update)
    {
        free_cross_product_cache();
        check(cond_M != nullptr, "cond_M must be set");
        check(dim_M == instruments.size(), "number of moments should be consistent");

        UInt n_instrument = instruments[0].size();
        for (const auto & x : instruments)
        {
            check(x.size() == n_instrument,
                  "number of instruments should be the same over all conditional moments.");
        }
        M_instrument_prod = new FLOAT[dim_M * n_instrument * sample_size];
        mean_M_instrument_prod = new FLOAT[dim_M * n_instrument];
        var_M_instrument_prod = new FLOAT[dim_M * n_instrument];

#pragma omp parallel for
        for (UInt i = 0; i < dim_M * n_instrument; ++i)
        {
            UInt m_idx = i % dim_M;
            UInt i_idx = i / dim_M;
            FLOAT * prod = M_instrument_prod + i * sample_size;
            const FLOAT * M = cond_M + m_idx * sample_size;
            const FLOAT * Ins = instruments[m_idx][i_idx];
#pragma omp simd
            for (UInt j = 0; j < sample_size; ++j)
            {
                prod[j] = M[j] * Ins[j];
            }
        }

        Basic_Statistics stat_of_products(M_instrument_prod, sample_size, dim_M * n_instrument);
        stat_of_products.mean(mean_M_instrument_prod);
        stat_of_products.var(var_M_instrument_prod);

        product_cache_need_update = false;
    }
}

