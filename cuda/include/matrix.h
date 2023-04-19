#pragma once
#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>

namespace matrix {
    float* allocateEmptyMatrixCuda(const int len);
    float* allocateEmptyMatrixCpu(const int len);
    float* generateRandomMatrixCuda(const int len);
}