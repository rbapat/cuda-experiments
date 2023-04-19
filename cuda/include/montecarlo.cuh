#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace montecarlo {
    __global__ void estimationKernel(curandState* states, int* blocksInCircle);
    float estimate_pi(int numSamples, int tpb);
    float time_pi_estimate(int numReps, int numSamples, int tpb);
}