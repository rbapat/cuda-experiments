#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <initializer_list>
#include <iostream>

#include "cudacommon.h"

namespace ml {
class Bias {
 public:
  Bias(cublasHandle_t cublasHandle, int numChannels);

  void fwd(cudnnHandle_t handle, cudnnTensorDescriptor_t outDesc, void* devOut);

  void sgd_bias(void* subtract, float learningRate);

  ~Bias();

 private:
  static constexpr float alpha = 1.0f;
  static constexpr float beta = 1.0f;

  cudnnTensorDescriptor_t inDesc;
  cublasHandle_t cublasHandle;
  int inDimA[4];
  int inStrideA[4];

  int numChannels;
  void* devBias;
};
}  // namespace ml