#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <initializer_list>
#include <iostream>

#include "cudacommon.h"

namespace ml {
class Pool2d {
 public:
  Pool2d(int batchSize, int numChannels, int rows, int cols, int kernelSize,
         int padding, int stride);

  void fwd(cudnnHandle_t handle, void* poolInput, void* poolOutput);
  void bwdData(cudnnHandle_t handle, void* poolInput, void* poolOutput,
               void* dLdPool, void* dLdIn);

  ~Pool2d();

 private:
  static constexpr int POOL_DIM = 2;
  static constexpr float alpha = 1.0f;
  static constexpr float beta = 0.0f;

  cudnnPoolingDescriptor_t poolDesc;
  int windowDimA[POOL_DIM];
  int paddingA[POOL_DIM];
  int strideA[POOL_DIM];

  cudnnTensorDescriptor_t inDesc;
  int inDimA[POOL_DIM + 2];
  int inStrideA[POOL_DIM + 2];

  cudnnTensorDescriptor_t outDesc;
  int outDimA[POOL_DIM + 2];
  int outStrideA[POOL_DIM + 2];
};

}  // namespace ml