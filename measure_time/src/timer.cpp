#include "../include/timer.h"

using namespace timer;
using namespace std;

void multi_timer::count_time(const string &activity)
{
    double duration = time_elapsed(start_timer);
    if (timers.find(activity) == timers.end())
    {
        timers.insert({activity, duration});
    } else
    {
        timers[activity] += duration;
    }

    start_timer = std::chrono::steady_clock::now();
}

void multi_timer::reset_timer()
{
    start_timer = std::chrono::steady_clock::now();
}

void multi_timer::print() const
{
    for ( const auto& key_value : timers )
    {
        std::cout << key_value.first << ": \t" << key_value.second << " seconds" << std::endl;
    }
}

multi_timer::multi_timer()
    : start_timer(now())
{}
