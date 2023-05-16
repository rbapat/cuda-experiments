#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <initializer_list>
#include <iostream>

#include "cudacommon.h"

namespace ml {
class Conv2d {
 public:
  Conv2d(cublasHandle_t cublasHandle, const int batchSize, const int rows,
         const int cols, const int outChannels, const int inChannels,
         const int kernelSize, const int stride, const int pad);

  void fwd(cudnnHandle_t handle, void* convInput, void* convOutput,
           void* workspace, size_t workspaceSize);
  void bwdFilter(cudnnHandle_t handle, void* convInput, void* dLdConv,
                 void* dLdW, void* workspace, size_t workspaceSize);
  void bwdData(cudnnHandle_t handle, void* convInput, void* dLdConv,
               void* dLdIn, void* workspace, size_t workspaceSize);

  void sgd_weight(void* subtract, float learningRate);

  size_t getWorkspaceSize(cudnnHandle_t handle);

  ~Conv2d();

 private:
  static constexpr int CONV_DIM = 2;
  static constexpr float alpha = 1.0f;
  static constexpr float beta = 0.0f;

  cublasHandle_t cublasHandle;

  cudnnConvolutionFwdAlgo_t fwdAlgo;
  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
  cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;

  cudnnConvolutionDescriptor_t convDesc;
  int padA[CONV_DIM];
  int filterStrideA[CONV_DIM];
  int dilationA[CONV_DIM];

  cudnnTensorDescriptor_t inDesc;
  int inDimA[CONV_DIM + 2];
  int inStrideA[CONV_DIM + 2];

  cudnnTensorDescriptor_t outDesc;
  int outDimA[CONV_DIM + 2];
  int outStrideA[CONV_DIM + 2];

  cudnnFilterDescriptor_t filterDesc;
  int filterDimA[CONV_DIM + 2];
  void* devFilter;
  int filterSize;
};

}  // namespace ml