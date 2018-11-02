#include <iostream>
#include "conditional_moment_ineq.h"
#include <Eigen/Dense>
using namespace std;
using namespace conditional_moment_inequality;

using Mat = Eigen::Matrix<FLOAT, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using Vec = Eigen::Matrix<FLOAT, Eigen::Dynamic, 1>;


template<typename T>
void
apply(Mat& M, T& f)
{
    FLOAT* M_ptr = M.data();
    UInt   n_M   = M.size();

    for (UInt i = 0; i < n_M; ++i)
    {
        M_ptr[i] = f(M_ptr[i]);
    }
}

// randn
inline void
randn(Mat& M, std::mt19937& rng)
{
    std::normal_distribution<double> normal_dist(0, 1);

    auto f = [&rng, &normal_dist](FLOAT x) { return normal_dist(rng); };
    apply(M, f);
}

int main()
{
    mt19937 rng(1000);
    if (sizeof(FLOAT) == 4)
    {
        cout << "runing with 32-bit float" << endl;
    }
    if (sizeof(FLOAT) == 8) cout << "runing with 64-bit FLOAT" << endl;

    int sample_size = 1000;
    Mat M(sample_size, 4);
    randn(M, rng);

    vector<FLOAT *> X(3);
    for (auto & x : X)
    {
        x = new FLOAT[sample_size];
        std::normal_distribution<double> normal_dist(0, 1);
        for (UInt i = 0; i < sample_size; ++i) x[i] = normal_dist(rng);
    }

    // Test true parameter
    cout << "Test true parameter: " << endl;
    Conditional_Moment_Ineq ineq(sample_size, 0.05, Conditional_Moment_Ineq::KS_S1);
    vector<int> sign(M.cols(), 0);
    ineq.set_cond_M(M.data(), M.cols(), sign.data());
    ineq.set_instruments(X, M.cols());

    bool test_result = ineq.inference();
    if (test_result)
    {
        cout << "the moment model is accepted." << endl;
    }
    else
    {
        cout << "the moment model is rejected." << endl;
    }

    // Test wrong parameter
    cout << "Test wrong parameter: " << endl;
    Mat M_wrong = M;
    M_wrong.col(0).array() -= 0.2;
    sign[0] = 1;
    ineq.set_cond_M(M_wrong.data(), M_wrong.cols(), sign.data());
    test_result = ineq.inference();
    if (test_result)
    {
        cout << "the moment model is accepted." << endl;
    }
    else
    {
        cout << "the moment model is rejected." << endl;
    }


    for (auto & x : X)
    {
        delete [] x;
    }
    return 0;
}
