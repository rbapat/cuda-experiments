#include "matrix.h"

namespace matrix {
float* allocateEmptyMatrixCuda(const int len) {
  float* ptr = nullptr;
  cudaError_t res = cudaMalloc(&ptr, len * len * sizeof(float));

  if (res != cudaSuccess) {
    throw std::runtime_error("Unable to allocate memory for empty cuda matrix");
  }

  return ptr;
}

float* allocateEmptyMatrixCpu(const int len) {
  return (float*)calloc(len * len, sizeof(float));
}

/*
Let's assume matrices here are row-major, so the matrix:
A B C
D E F
G H I
is represented as `A B C D E F G H I` in memory
*/
float* generateRandomMatrixCuda(const int len) {
  float* cpuMat = allocateEmptyMatrixCpu(len);
  for (int i = 0; i < len * len; i++) {
    cpuMat[i] = (float)rand() / RAND_MAX;
  }

  float* gpuMat = allocateEmptyMatrixCuda(len);
  cudaError_t res = cudaMemcpy(gpuMat, cpuMat, len * len * sizeof(float),
                               cudaMemcpyHostToDevice);

  free(cpuMat);
  if (res != cudaSuccess) {
    throw std::runtime_error(
        "Failed to copy randomly allocated matrix from host to device");
  }

  return gpuMat;
}
}  // namespace matrix