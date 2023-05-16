#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <initializer_list>
#include <iostream>

#include "cudacommon.h"

namespace ml {
class ReLU {
 public:
  ReLU();

  void fwd(cudnnHandle_t handle, cudnnTensorDescriptor_t tensorDesc, void* in,
           void* out);
  void bwdData(cudnnHandle_t handle, cudnnTensorDescriptor_t tensorDesc,
               void* in, void* out, void* dLdRelu, void* dLdIn);

  ~ReLU();

 private:
  cudnnActivationDescriptor_t reluDesc;
  static constexpr float alpha = 1.0f;
  static constexpr float beta = 0.0f;
};

}  // namespace ml