#include "conditional_moment_ineq.h"
#include "eigen_notation.h"

using namespace conditional_moment_inequality;

void Conditional_Moment_Ineq::prepare_GMS() const
{
    if (GMS_need_update)
    {
        free_GMS();
        update_cross_product_cache();
        kappa = sqrt(0.3 * log(sample_size));
        B = sqrt(0.4 * log(sample_size) / log(log(sample_size)));

        // first calculate xi
        FLOAT xi_constant = sqrt_sample_size / kappa;

        UInt dim_ins = instruments[0].size();
        Mat xi = MapMat(var_M_instrument_prod, dim_M, dim_ins);
        xi.colwise() += epsilon * MapVec(var_M, dim_M);
        xi =  xi_constant * MapMat(mean_M_instrument_prod, dim_M, dim_ins).array() / xi.array().sqrt();

        // next, transform xi to GMS (i.e. D * phi in Andrews and Shi(2013))
        GMS = new FLOAT[dim_M * dim_ins];
        const FLOAT * xi_ptr = xi.data();

    #pragma omp parallel for
        for (UInt i = 0; i < dim_M * dim_ins; ++i)
        {
            int sign = this->sign[i % dim_M];
            if (sign == 1)
            {
                GMS[i] = (xi_ptr[i] > 1) ? B : 0.0;
            }
            if (sign == -1)
            {
                GMS[i] = (xi_ptr[i] < -1) ? -B : 0.0;
            }
            if (sign == 0)
            {
                GMS[i] = 0;
            }
        }
        MapMat GMS_mat(GMS, dim_M, dim_ins);
        GMS_mat.array().colwise() *= MapVec(var_M, dim_M).array().sqrt();

        GMS_need_update = false;
    }
}
