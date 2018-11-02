# GPU-CMI
 a GPU-Accelerated Implementation of Conditional Moment Inequality Inference 

GPU-CMI is a GPU-Accelerated implementation of Andrews and Shi(2013). This implementation exploits the parallel feature in the inference procedure and utilizes GPU computation whenever it is possible. This makes it much faster than the original implementation. Moreover, it allows users to implement custom instrumental functions in addition to the hypercube family in the original paper. The library is written in C++ and CUDA. Its source code is published under the Apache License 2.0.
