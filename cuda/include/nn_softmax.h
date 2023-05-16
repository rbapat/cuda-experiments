#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <initializer_list>
#include <iostream>

#include "cudacommon.h"

namespace ml {
class Softmax {
 public:
  Softmax(cudnnHandle_t cudnnHandle, int batchSize, int numPreds);

  void fwd(void* in, void* out);
  void bwdData(void* in, void* out, void* dLdOut, void* dLdIn);

  ~Softmax();

 private:
  static constexpr float alpha = 1.0f;
  static constexpr float beta = 0.0f;

  cudnnHandle_t cudnnHandle;
  cudnnTensorDescriptor_t desc;
  int inDimA[4];
  int inStrideA[4];

  int batchSize, numPreds;
};

}  // namespace ml