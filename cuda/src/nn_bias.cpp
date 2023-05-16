#include "nn_bias.h"

namespace ml {
Bias::Bias(cublasHandle_t cublasHandle, int numChannels)
    : cublasHandle(cublasHandle), numChannels(numChannels) {
  cudnnCheckError(cudnnCreateTensorDescriptor(&inDesc));

  ml::initStrides(inDimA, inStrideA, 1, numChannels, 1, 1);
  cudnnCheckError(cudnnSetTensorNdDescriptor(inDesc, CUDNN_DATA_FLOAT, 4,
                                             inDimA, inStrideA));

  cudaCheckError(cudaMalloc(&devBias, numChannels * sizeof(float)));
  cudaCheckError(cudaMemset(devBias, 0, numChannels * sizeof(float)));
}

void Bias::fwd(cudnnHandle_t handle, cudnnTensorDescriptor_t outDesc,
               void* devOut) {
  cudnnCheckError(
      cudnnAddTensor(handle, &alpha, inDesc, devBias, &beta, outDesc, devOut))
}

void Bias::sgd_bias(void* subtract, float learningRate) {
  float updateAlpha = -learningRate;
  cublasCheckError(cublasSaxpy_v2(cublasHandle, numChannels, &updateAlpha,
                                  (const float*)subtract, 1, (float*)devBias,
                                  1));
}

Bias::~Bias() { cudnnCheckError(cudnnDestroyTensorDescriptor(inDesc)); }
}  // namespace ml