#include <iostream>
#include <cmath>
#include "../include/timer.h"

using namespace std;


int main()
{
    size_t n_loop = 100000000;
    double a = 0;

    auto start_timer = timer::now();

    for(size_t i = 0; i < n_loop; ++i){
        a = sin(i);
    }

    double dur_time = timer::time_elapsed(start_timer);

    cout << "elapsed time: " << dur_time << " seconds" << endl;

    return 0;
}

