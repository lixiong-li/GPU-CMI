#pragma once
#include <Eigen/Dense>
#include <set>
#include <iostream>
#include <vector>

namespace QuadProg {

#ifdef USE_32_BIT_FLOAT
    using FLOAT= float;
#else
    using FLOAT = double;
#endif
using UInt = uint_fast32_t;

using Mat = Eigen::Matrix<FLOAT, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using Vec = Eigen::Matrix<FLOAT, Eigen::Dynamic, 1>;

using MapMat          = Eigen::Map<Mat>;
using ConstMapMat     = Eigen::Map<const Mat>;
using MapVec          = Eigen::Map<Vec>;
using ConstMapVec     = Eigen::Map<const Vec>;

using std::vector;

// min 0.5 * x' G x + c * x
// s.t. x[i] >= 0 if sign[i] == 1
//      x[i] =  0 if sign[i] == 0
//      x[i] <= 0 if sign[i] == -1
// Using active set algorithm

inline
FLOAT quadprog(const Mat & G, const Vec & c, const vector<int> & sign)
{
    UInt dx = c.size();
    assert(G.cols() == dx and G.rows() == dx);

    Vec x = Vec::Zero(dx);
    std::set<UInt> W;
    std::set<UInt> WC; // complement of W
    for (UInt i = 0; i < dx; ++i)
    {
        W.insert(i);
    }


    constexpr FLOAT eps = sizeof(FLOAT) == 4 ? 1e-4 : 1e-8;

    if (W.size() > 0)
    {
        for (UInt k = 0; k < 1000000; ++k)
        {
            // solve p
            Vec g = G * x + c;
            Vec p = Vec::Zero(dx);
            if (W.size() < dx)
            {
                Mat G_t(WC.size(), WC.size());
                Vec g_t(WC.size());

                UInt row_idx = 0;
                for (UInt i : WC)
                {
                    g_t[row_idx] = g[i];
                    UInt col_idx = 0;
                    for (UInt j : WC)
                    {
                        G_t(row_idx, col_idx) = G(i,j);
                        ++col_idx;
                    }
                    ++row_idx;
                }
                Vec p_t = -G_t.llt().solve(g_t);

                row_idx = 0;
                for (UInt i : WC)
                {
                    p[i] = p_t[row_idx];
                    ++row_idx;
                }
            }

            if (p.transpose() * p < eps)
            {
                // if p == 0

                Vec lambda = G * x + c;
                for (UInt i = 0; i < dx; ++i)
                {
                    if (sign[i] == 0) lambda[i] = 0;
                    if (sign[i] == -1) lambda[i] *= -1;
                }
                bool converge = true;
                UInt idx_to_be_removed = dx;
                FLOAT min_lambda = -eps;

                for (UInt i : W)
                {                    
                    converge &= lambda[i] >= -eps;
                    if (lambda[i] < min_lambda)
                    {
                        min_lambda = lambda[i];
                        idx_to_be_removed = i;
                    }
                }
                if (converge) break;
                assert(idx_to_be_removed != dx);
                W.erase(idx_to_be_removed);
                WC.insert(idx_to_be_removed);
            }
            else
            {
                FLOAT alpha = 1;
                UInt add_idx = dx;
                for (UInt i : WC)
                {
                    if (p[i] * sign[i] < 0)
                    {
                        if (-x[i] / p[i] < alpha)
                        {
                            alpha = -x[i] / p[i];
                            add_idx = i;
                        }
                    }
                }
                x.array() += alpha * p.array();

                if (add_idx != dx)
                {
                    W.insert(add_idx);
                    WC.erase(add_idx);
                }
            }

            assert(!x.hasNaN());
            assert(x.allFinite());
        }
    }

    return x.transpose() * (0.5 * G * x + c);

}













}
