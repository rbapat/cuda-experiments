#include "matrix.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace matmul {

    namespace naive {
        __global__ void matmul_kernel(float* mat1, float* mat2, float *out, int matrixSize);
        float time_execution(int num_times, int matrixSize, int tpb);
    }

    namespace cublas {
        float time_execution(int numTimes, int matrixSize, int tpb);
    }
}