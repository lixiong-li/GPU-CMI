#include "basic_statistics.h"
#include "conditional_moment_ineq.h"
#include <assert.h>
#include <limits>
#include "eigen_notation.h"

using namespace conditional_moment_inequality;


using basic_statistics::Basic_Statistics;

pair<UInt, UInt>
choose_grids(UInt min_sample_in_cell, UInt n_active_X, UInt sample_size)
{
    UInt n_grid_smallest = 2;
    UInt n_grid_largest  = 2;

    FLOAT grid_upper_bound = 2 + pow(1.0 * sample_size / min_sample_in_cell, 1.0 / n_active_X);

    double size_limit = 8.0 * 8 * 1024 * 1024 * 1024;

    grid_upper_bound =
      std::min((double)grid_upper_bound, pow(size_limit / sample_size, 1.0 / n_active_X));

    FLOAT min_gap = std::numeric_limits<FLOAT>::infinity();
    for (UInt i = 2; i <= grid_upper_bound; i += 2)
    {
        FLOAT gap = abs(min_sample_in_cell - sample_size / pow(i, n_active_X));
        if (gap < min_gap)
        {
            min_gap        = gap;
            n_grid_largest = i;
        }
    }

    // check if generated results makes sense
    assert(n_grid_smallest >= 2);
    assert(n_grid_smallest % 2 == 0);
    assert(n_grid_largest >= n_grid_smallest);
    assert(n_grid_largest % 2 == 0);
    assert(pow((double)n_grid_largest, (double)n_active_X) < std::numeric_limits<UInt>::max());

    //std::cout << "n_grid_largest " << n_grid_largest << std::endl;

    return {n_grid_smallest, n_grid_largest};
}

inline UInt
power_UInt(UInt base, UInt order)
{
    UInt result = 1;
    for (UInt i = 0; i < order; ++i)
    {
        result *= base;
    }
    return result;
}

// -----------------------------
// On combitorials

// calculate binomial coefficients
// number of possible combinations when choosing k numbers out of n numbers;
inline UInt
binomial_coeff(UInt n, UInt k)
{
    assert(n >= k);
    if (n == k || k == 0)
    {
        return 1;
    }

    if (k > n - k)
    {
        k = n - k;
    }

    return binomial_coeff(n - 1, k - 1) + binomial_coeff(n - 1, k);
}

// ---------------------------------------------------------
// unique combination: enumerate {v: v[i] in {0, 1, ..., n-1} and v[i] != v[j] for any i, j = 0, 1,
// ..., k-1}
inline std::vector<std::vector<UInt>>
unique_comb(UInt n, UInt k)
{
    assert(k <= n);
    assert(k >= 1);

    UInt                 n_A = binomial_coeff(n, k);
    vector<vector<UInt>> A;
    A.reserve(n_A);

    // initialize x = {0, 1, ..., n-1}
    vector<UInt> x(n);
    for (UInt i = 0; i < n; ++i)
        x[i] = i;

    while (A.size() < n_A)
    {
        // let x_first_k contain the first k elements of x, and then sort it
        vector<UInt> x_first_k(x.cbegin(), x.cbegin() + k);
        sort(x_first_k.begin(), x_first_k.end());

        // check if xx has been included in A
        bool already_included = false;
        for (const auto& a : A)
        {
            if (a == x_first_k)
            {
                already_included = true;
                break;
            }
        }
        if (!already_included)
            A.push_back(x_first_k);

        // go on and permute x
        next_permutation(x.begin(), x.end());
    }
    return A;
}

pair<FLOAT, FLOAT>
extreme(const FLOAT* v, UInt length)
{
    FLOAT min_value = std::numeric_limits<FLOAT>::infinity();
    FLOAT max_value = -std::numeric_limits<FLOAT>::infinity();
    for (UInt i = 0; i < length; ++i)
    {
        min_value = std::min(min_value, v[i]);
        max_value = std::max(max_value, v[i]);
    }
    return {min_value, max_value};
}

pair<FLOAT, FLOAT>
extreme(const vector<const FLOAT*>& X, UInt length)
{
    FLOAT min_value = std::numeric_limits<FLOAT>::infinity();
    FLOAT max_value = -std::numeric_limits<FLOAT>::infinity();
    for (const auto& x : X)
    {
        auto temp = extreme(x, length);
        min_value = std::min(min_value, temp.first);
        max_value = std::max(max_value, temp.second);
    }
    return {min_value, max_value};
}

// ------------------------------------------------------------
// compuate average and standard error of a vector
// ------------------------------------------------------------

// Calculate the inverse square root of matrix
// i.e. given A, return B so that B * B = the inverse of A
template<class Matrix>
inline Mat
inv_square_root(const Matrix& M)
{
    Eigen::SelfAdjointEigenSolver<Mat> eigensolver(M);
    assert(eigensolver.info() == Eigen::Success);
    Mat result = eigensolver.operatorInverseSqrt();
    if (result.hasNaN())
    {
        throw std::runtime_error("the resulted matrix has NaN!");
    }
    return result;
}

// cdf of normal distribution
inline FLOAT
normal_cdf(FLOAT value)
{
    constexpr FLOAT inv_sqrt2 = 0.70710678118; // 1.0 / sqrt(2)
    return 0.5 * erfc(-value * inv_sqrt2);
}

inline void
normal_cdf(FLOAT* value, UInt length)
{
#pragma omp parallel for schedule(static)
    for (UInt i = 0; i < length; ++i)
    {
        value[i] = normal_cdf(value[i]);
    }
}

void
transform_X(Mat& X)
{
    // transform X
    Basic_Statistics X_statistics(X);
    UInt             dim_X = X.cols();
    Vec              mean_X(dim_X);
    Mat              cov_X(dim_X, dim_X);
    Vec              var_X(dim_X);
    X_statistics.cov(cov_X);
    X_statistics.var(var_X);
    X_statistics.mean(mean_X);

    X.rowwise() -= mean_X.transpose();
    try
    {
        X = X * inv_square_root(cov_X);
    }
    catch (runtime_error err)
    {
        var_X = var_X.array().cwiseSqrt().cwiseInverse();
        X     = X * var_X.asDiagonal();
    }
    normal_cdf(X.data(), X.size());
}

Mat
transform_X(const vector<const FLOAT*>& X, UInt length)
{
    UInt dim_X = X.size();
    Mat  X_mat(length, dim_X);
    for (UInt k = 0; k < dim_X; ++k)
    {
        FLOAT* X_ = X_mat.data() + k * length;
        for (UInt i = 0; i < length; ++i)
            X_[i] = X[k][i];
    }
    transform_X(X_mat);
    return X_mat;
}

inline FLOAT weight_func(UInt n_grid)
{
   assert(n_grid % 2 == 0);
   UInt r = n_grid / 2;
   return 1.0 / (r * r + 100);
}

pair<vector<const FLOAT *>, vector<FLOAT> > conditional_moment_inequality::cube_instrument(const vector<const FLOAT*>& X, UInt sample_size)
{
    for (const auto & x : X) check(x != nullptr, "instrument should be provided");

    UInt dim_X = X.size();
    // when generating box instruments, we only use 'n_active_X' instruments at a time,
    // although the identity of those 'active_X' will loop over all possible combinations
    UInt n_active_X = std::min((UInt)3, (UInt)dim_X);
    // when generating box instruments for one particular set of 'active_X',
    // we put uniform grid on [0, 1], the number of grid ranging from 'n_grid_low' to 'n_grid_up'
    // the 'n_grid_up' is chosen to ensure the expected number of samples in each cell is at least
    // 'min_sample_in_cell'
    UInt min_sample_in_cell = 40;
    // In Andrews and Shi(2013), the number of grids is always even number. I follow their
    // settings.

    UInt n_grid_smallest, n_grid_largest;
    tie(n_grid_smallest, n_grid_largest) =
      choose_grids(min_sample_in_cell, n_active_X, sample_size);

    // when generating box instruments, we only use 'n_active_X' instruments at a time,
    // although the identity of those 'active_X' will loop over all possible combinations
    UInt n_possible_combinations               = binomial_coeff(dim_X, n_active_X);
    auto possbile_combinations_of_active_X_idx = unique_comb(dim_X, n_active_X);

    // Now, calculate the number of total instruments:
    // Note: In Andrews and Shi(2013), the number of grids is always even number.
    // I follow their settings.
    //
    UInt n_boxs = 0; // number of instruments given one set of active X
    for (UInt n_grid = n_grid_smallest; n_grid <= n_grid_largest; n_grid += 2)
    {
        n_boxs += power_UInt(n_grid, n_active_X);
    }
    // number of total instruments
    UInt n_instruments = n_possible_combinations * n_boxs;

    // transform instrument into [0, 1] cube if necessary
    vector<const FLOAT*> X_transformed(dim_X);
    auto           min_max             = extreme(X, sample_size);
    bool           need_transformation = min_max.first < 0 or min_max.second > 1;
    Mat            X_temp;
    if (need_transformation)
    {
        X_temp = transform_X(X, sample_size);
        for (UInt k = 0; k < dim_X; ++k)
            X_transformed[k] = X_temp.data() + k * sample_size;
    }
    else
    {
        X_transformed = X;
    }

    // Now, compute all those instruments and weights
    //
    vector<FLOAT*> instruments(n_instruments);
    vector<FLOAT>  weights(n_instruments, 0);

    // initialization
    for (UInt xi = 0; xi < n_instruments; ++xi)
    {
        instruments[xi] = new FLOAT[sample_size]();
        for (UInt i = 0; i < sample_size; ++i) instruments[xi][i] = 1;
    }

    // loop over all possible combinations of active X
#pragma omp parallel for
    for (UInt j = 0; j < n_possible_combinations; ++j)
    {
        UInt instrument_idx = j * n_boxs;
        // jth possible combination
        const auto& active_X_idx = possbile_combinations_of_active_X_idx[j];

        // loop over all possible boxes
        for (UInt n_grid = n_grid_smallest; n_grid <= n_grid_largest; n_grid += 2)
        {
            UInt        n_instruments_with_same_grid_size = power_UInt(n_grid, n_active_X);

            for (UInt grids_idx = 0; grids_idx < power_UInt(n_grid, n_active_X); ++grids_idx)
            {
                FLOAT* instrument_ptr = instruments[instrument_idx];
                UInt idx = grids_idx;
                for (UInt xi = 0; xi < n_active_X; ++xi)
                {
                    UInt which_grid = idx % n_grid;
                    idx = idx / n_grid;
                    const FLOAT* X_ptr = X_transformed[active_X_idx[xi]];
                    for (UInt i = 0; i < sample_size; ++i)
                    {
                        FLOAT t = n_grid * X_ptr[i];
                        UInt t_int = floor(t);
                        instrument_ptr[i] *= (FLOAT)(which_grid == t_int or (t_int == n_grid and which_grid + 1 == n_grid));
                    }
                }
                weights[instrument_idx] = weight_func(n_grid) / n_instruments_with_same_grid_size;
                ++instrument_idx;
            }
        }
    }

    vector<const FLOAT*> const_instruments(n_instruments);
    for (UInt i = 0; i < n_instruments; ++i)
    {
        const_instruments[i] = instruments[i];
    }

    return {const_instruments, weights};
}
