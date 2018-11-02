#pragma once

#ifndef TIMER_v1_0_H
#define TIMER_v1_0_H

#include <chrono>
#include <vector>
#include <string>
#include <map>
#include <iostream>

namespace timer{
//using namespace std;

inline std::chrono::steady_clock::time_point
now()
{
    return std::chrono::steady_clock::now();
}

template<typename T>
double duration_in_secs(T duration)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 1000.0 ;
}

inline
double time_elapsed(std::chrono::steady_clock::time_point & old_time)
{
    // return elapsed time with respect to 'old_time'
    auto current_time = now();
    return duration_in_secs(current_time - old_time);
}



class multi_timer
{
    std::map<std::string, double> timers;
    std::chrono::steady_clock::time_point start_timer;

public:
    multi_timer();

    void reset_timer();

    void count_time(const std::string & activity);

    void print() const;
};

}



#endif // TIMER_v1_0_H
