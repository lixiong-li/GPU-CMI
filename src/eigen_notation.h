#pragma once

#include "conditional_moment_ineq.h"
#include <Eigen/Dense>
namespace conditional_moment_inequality {

using Mat = Eigen::Matrix<FLOAT, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using Vec = Eigen::Matrix<FLOAT, Eigen::Dynamic, 1>;

using MapMat          = Eigen::Map<Mat>;
using ConstMapMat     = Eigen::Map<const Mat>;
using MapVec          = Eigen::Map<Vec>;
using ConstMapVec     = Eigen::Map<const Vec>;




}
