#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <initializer_list>
#include <iostream>

#include "cudacommon.h"

namespace ml {
class Linear {
 public:
  Linear(cublasHandle_t cublasHandle, int batchSize, int inChannels,
         int outChannels);

  void fwd(void* in, void* out);
  void bwdW(void* in, void* dLdMul, void* dLdW);
  void bwdData(void* dLdMul, void* dLdW);

  void sgd_weight(void* subtract, float learningRate);

  ~Linear();

 private:
  static constexpr float alpha = 1.0f;
  static constexpr float beta = 0.0f;

  cublasHandle_t cublasHandle;
  void* devWeights;

  int batchSize, inChannels, outChannels;
};

}  // namespace ml