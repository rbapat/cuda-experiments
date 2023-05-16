#include "nn_linear.h"

namespace ml {

Linear::Linear(cublasHandle_t cublasHandle, int batchSize, int inChannels,
               int outChannels)
    : cublasHandle(cublasHandle),
      batchSize(batchSize),
      inChannels(inChannels),
      outChannels(outChannels) {
  cublasCheckError(cublasCreate_v2(&cublasHandle));

  cudaCheckError(
      cudaMalloc(&devWeights, inChannels * outChannels * sizeof(float)));
  cudaCheckError(
      cudaMemset(devWeights, 0, inChannels * outChannels * sizeof(float)));
}

void Linear::fwd(void* in, void* out) {
  cublasCheckError(cublasSgemm_v2(
      cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, outChannels, batchSize,
      inChannels, &alpha, (const float*)in, outChannels,
      (const float*)devWeights, inChannels, &beta, (float*)out, outChannels));
}

// dLdMul * x^T
void Linear::bwdW(void* in, void* dLdMul, void* dLdW) {
  cublasCheckError(cublasSgemm_v2(
      cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, outChannels, inChannels,
      batchSize, &alpha, (const float*)dLdMul, outChannels, (const float*)in,
      inChannels, &beta, (float*)dLdW, outChannels));
}

// w^T * dLdMul
void Linear::bwdData(void* dLdMul, void* dLdW) {
  cublasCheckError(cublasSgemm_v2(
      cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, inChannels, batchSize,
      outChannels, &alpha, (const float*)devWeights, outChannels,
      (const float*)dLdMul, outChannels, &beta, (float*)dLdW, inChannels));
}

void Linear::sgd_weight(void* subtract, float learningRate) {
  float updateAlpha = -learningRate;
  cublasCheckError(cublasSaxpy_v2(
      cublasHandle, inChannels * outChannels * sizeof(float), &updateAlpha,
      (const float*)subtract, 1, (float*)devWeights, 1));
}

Linear::~Linear() {}
}  // namespace ml
