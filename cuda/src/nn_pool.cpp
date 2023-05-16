#include "nn_pool.h"

namespace ml {

Pool2d::Pool2d(int batchSize, int numChannels, int rows, int cols,
               int kernelSize, int padding, int stride)
    : windowDimA{kernelSize, kernelSize},
      paddingA{padding, padding},
      strideA{stride, stride} {
  cudnnCheckError(cudnnCreatePoolingDescriptor(&poolDesc));
  cudnnCheckError(cudnnCreateTensorDescriptor(&inDesc));
  cudnnCheckError(cudnnCreateTensorDescriptor(&outDesc));

  // TODO: move the rest of this into the fwd/bwd functions
  cudnnSetPoolingNdDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                              POOL_DIM, windowDimA, paddingA, strideA);

  ml::initStrides(inDimA, inStrideA, batchSize, numChannels, rows, cols);
  cudnnCheckError(cudnnSetTensorNdDescriptor(inDesc, CUDNN_DATA_FLOAT,
                                             POOL_DIM + 2, inDimA, inStrideA));

  ml::initStrides(outDimA, outStrideA, batchSize, numChannels,
                  rows / kernelSize, cols / kernelSize);
  cudnnCheckError(cudnnSetTensorNdDescriptor(
      outDesc, CUDNN_DATA_FLOAT, POOL_DIM + 2, outDimA, outStrideA));

  cudnnCheckError(cudnnGetPoolingNdForwardOutputDim(poolDesc, inDesc,
                                                    POOL_DIM + 2, outDimA));

  ml::initStrides(outDimA, outStrideA, outDimA[0], outDimA[1], outDimA[2],
                  outDimA[3]);  // theres prob some c++ way to unpack outDimA
  cudnnCheckError(cudnnSetTensorNdDescriptor(
      inDesc, CUDNN_DATA_FLOAT, POOL_DIM + 2, outDimA, outStrideA));
}

void Pool2d::fwd(cudnnHandle_t handle, void* poolInput, void* poolOutput) {
  cudnnCheckError(cudnnPoolingForward(handle, poolDesc, &alpha, inDesc,
                                      poolInput, &beta, outDesc, poolOutput));
}
void Pool2d::bwdData(cudnnHandle_t handle, void* poolInput, void* poolOutput,
                     void* dLdPool, void* dLdIn) {
  cudnnCheckError(cudnnPoolingBackward(handle, poolDesc, &alpha, outDesc,
                                       poolOutput, outDesc, dLdPool, inDesc,
                                       poolInput, &beta, inDesc, dLdIn));
}

Pool2d::~Pool2d() {
  cudnnCheckError(cudnnDestroyPoolingDescriptor(poolDesc));
  cudnnCheckError(cudnnDestroyTensorDescriptor(inDesc));
  cudnnCheckError(cudnnDestroyTensorDescriptor(outDesc));
}

}  // namespace ml