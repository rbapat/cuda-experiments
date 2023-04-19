#include "matrix.h"

namespace matmul {

    namespace naive {
        __global__ void matmul_kernel(float* mat1, float* mat2, float *out, int matrixSize);
        float time_execution(int num_times, int matrixSize, int tpb);
    }
}