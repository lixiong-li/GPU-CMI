#pragma once

#ifndef NOTATION_H
#define NOTATION_H

#include <vector>
#include <tuple>
#include <random>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <utility>
#include <cmath>


namespace conditional_moment_inequality {

#ifdef USE_32_BIT_FLOAT
    using FLOAT= float;
#else
    using FLOAT = double;
#endif

using UInt = uint_fast32_t;
using Huge_UInt = uint_fast64_t;

using std::vector;
using std::pair;
using std::mt19937;
using std::tuple;
using std::string;
using std::runtime_error;
using std::make_tuple;
using std::make_pair;
using std::tie;


inline void delete_array(const FLOAT * pointer)
{
    if (pointer != nullptr)
    {
        delete [] pointer;
    }
}

inline void delete_array(FLOAT * pointer)
{
    if (pointer != nullptr)
    {
        delete [] pointer;
    }
}

inline
void check(bool should_be_true, string message)
{
    if (not should_be_true)
    {
        throw runtime_error(message);
    }
}



}









#endif
