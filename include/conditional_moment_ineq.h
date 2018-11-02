#pragma once

#include "notation.h"


namespace conditional_moment_inequality
{

pair<vector<const FLOAT*>, vector<FLOAT>> cube_instrument(const vector<const FLOAT*> & X,
                                                            UInt sample_size);


class Conditional_Moment_Ineq
{
    // E[M|X] >= 0
  public:
    // ------------------------------------------
    // INTERFACE
    // ------------------------------------------
    static constexpr int CvM_S1 = 1;
    static constexpr int KS_S1 = 2;
    static constexpr int CvM_S2 = 3;
    static constexpr int KS_S2 = 4;

    static constexpr int eq_zero = 0;
    static constexpr int greater_zero = 1;
    static constexpr int less_zero = -1;

    // constructor and destructor
    Conditional_Moment_Ineq(UInt sample_size,
                            FLOAT alpha = 0.05,
                            int stat_type = KS_S1,
                            UInt n_bootstrap = 2000,
                            FLOAT epsilon = 0.05);
    ~Conditional_Moment_Ineq();

    // provide data
    void set_cond_M(const FLOAT* cond_M, UInt dim_M, const int * sign = nullptr);
    void set_instruments(const vector<const FLOAT*>& X, UInt dim_M);
    void set_instruments(const vector<FLOAT*>& X, UInt dim_M);

    void set_instruments(const vector<vector<FLOAT*>>& X);
    void set_instruments(const vector<vector<FLOAT*>>& X, const vector<FLOAT>& weight);

    void set_instruments(const vector<vector<const FLOAT*>>& X);
    void set_instruments(const vector<vector<const FLOAT*>>& X, const vector<FLOAT>& weight);



    // inference
    bool inference() const;

    // p value
    FLOAT p_value() const;

    // test statistics
    FLOAT test_statistics() const;

  private:
    // ----------------------------------------
    // IMPLEMENTATION DETAILS
    // ----------------------------------------

    // Dimensions
    const UInt sample_size;
    const FLOAT sqrt_sample_size;
    UInt dim_M;
    vector<int> sign;

    // Data
    const FLOAT*         cond_M;
    vector<vector<const FLOAT*>> instruments;
    vector<FLOAT>          weight;

    bool own_instruments; // true if we manages the memory of instruments
    void free_instruments();

    // Criteria Function
    FLOAT S(const FLOAT* M, const FLOAT* Sigma_var, const FLOAT* Sigma_cov) const;

    // cache: cov of M
    mutable FLOAT* mean_M;
    mutable FLOAT* var_M;
    mutable FLOAT* cov_M;

    void free_M_cache() const;
    void update_M_cache() const;

    // cache: cross-product of M and instruments, and its mean and variance
    mutable bool product_cache_need_update;
    mutable FLOAT* M_instrument_prod;
    mutable FLOAT* mean_M_instrument_prod;
    mutable FLOAT* var_M_instrument_prod;

    void free_cross_product_cache() const;
    void update_cross_product_cache() const;

    // cache: Moment Selection (GMS)
    mutable bool GMS_need_update;
    mutable FLOAT B;
    mutable FLOAT kappa;
    const FLOAT epsilon;  // regularize covariance matrix
    mutable FLOAT* GMS;    // phi -> inf if E[M|X] > 0; phi -> 0 if E[M|X] = 0
    void prepare_GMS() const;
    void free_GMS() const;

    // Test Statistics and its critical value by Bootstrap
    FLOAT alpha; // 1 - alpha  is the size of the test, alpha should be less than 0.5
    int stat_type; // whether we use CvM (Cramér-von Mises) or KS (Kolmogorov-Smirnov) statistics

    const UInt n_bootstrap;

    vector<FLOAT*> bootstrap_multiplier;
    void free_bootstrap_multiplier();
    void gen_bootstrap_multiplier();

    static constexpr FLOAT inf = std::numeric_limits<FLOAT>::infinity();
    vector<FLOAT> boostrap_test_statistics(FLOAT x = inf) const;


};




}
