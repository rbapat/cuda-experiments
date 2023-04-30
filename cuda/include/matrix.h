#pragma once
#include <cuda_runtime.h>
#include <stdlib.h>

#include <iostream>

namespace matrix {
float* allocateEmptyMatrixCuda(const int len);
float* allocateEmptyMatrixCpu(const int len);
float* generateRandomMatrixCuda(const int len);
}  // namespace matrix