#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define cudaCheckError(err)                                    \
  {                                                            \
    if (err != cudaSuccess) {                                  \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                         \
      exit(0);                                                 \
    }                                                          \
  }

#define cudnnCheckError(err)                                    \
  {                                                             \
    if (err != CUDNN_STATUS_SUCCESS) {                          \
      printf("cuDNN failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudnnGetErrorString(err));                         \
      exit(0);                                                  \
    }                                                           \
  }

#define cublasCheckError(err)                                       \
  {                                                                 \
    if (err != CUBLAS_STATUS_SUCCESS) {                             \
      printf("cuBLAS failure %s:%d: '%s:%s'\n", __FILE__, __LINE__, \
             cublasGetStatusName(err), cublasGetStatusString(err)); \
      exit(0);                                                      \
    }                                                               \
  }

namespace ml {
template <typename... Args>
void initStrides(int* dims, int* strides, Args... args) {
  const size_t numDims = sizeof...(args);

  size_t index = 0;
  ((dims[index++] = static_cast<int>(args)), ...);

  strides[numDims - 1] = 1;
  for (int i = numDims - 2; i >= 0; i--) {
    strides[i] = dims[i + 1] * strides[i + 1];
  }
}

}  // namespace ml