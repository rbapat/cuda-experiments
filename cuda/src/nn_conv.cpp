#include "nn_conv.h"

namespace ml {
Conv2d::Conv2d(cublasHandle_t cublasHandle, const int batchSize, const int rows,
               const int cols, const int outChannels, const int inChannels,
               const int kernelSize, const int stride, const int pad)
    : padA{pad, pad},
      filterStrideA{stride, stride},
      dilationA{1, 1},
      filterDimA{outChannels, inChannels, kernelSize, kernelSize},
      cublasHandle(cublasHandle) {
  cudnnCheckError(cudnnCreateConvolutionDescriptor(&convDesc));
  cudnnCheckError(cudnnCreateFilterDescriptor(&filterDesc));
  cudnnCheckError(cudnnCreateTensorDescriptor(&inDesc));
  cudnnCheckError(cudnnCreateTensorDescriptor(&outDesc));

  filterSize =
      outChannels * inChannels * kernelSize * kernelSize * sizeof(float);
  cudaCheckError(cudaMalloc(&devFilter, filterSize));
  cudaCheckError(cudaMemset(devFilter, 0.f, filterSize));

  // TODO: move the rest of this into the fwd/bwd functions
  ml::initStrides(inDimA, inStrideA, batchSize, inChannels, rows, cols);
  cudnnCheckError(cudnnSetTensorNdDescriptor(inDesc, CUDNN_DATA_FLOAT,
                                             CONV_DIM + 2, inDimA, inStrideA));

  ml::initStrides(outDimA, outStrideA, batchSize, outChannels, rows, cols);
  cudnnCheckError(cudnnSetTensorNdDescriptor(
      outDesc, CUDNN_DATA_FLOAT, CONV_DIM + 2, outDimA, outStrideA));

  cudnnCheckError(cudnnSetConvolutionNdDescriptor(
      convDesc, CONV_DIM, padA, filterStrideA, dilationA, CUDNN_CONVOLUTION,
      CUDNN_DATA_FLOAT));

  cudnnCheckError(cudnnSetConvolutionMathType(convDesc, CUDNN_DEFAULT_MATH));

  cudnnCheckError(cudnnSetFilterNdDescriptor(filterDesc, CUDNN_DATA_FLOAT,
                                             CUDNN_TENSOR_NCHW, CONV_DIM + 2,
                                             filterDimA));

  fwdAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  bwdFilterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  bwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
}

void Conv2d::fwd(cudnnHandle_t handle, void* convInput, void* convOutput,
                 void* workspace, size_t workspaceSize) {
  cudnnCheckError(cudnnConvolutionForward(
      handle, &alpha, inDesc, convInput, filterDesc, devFilter, convDesc,
      fwdAlgo, workspace, workspaceSize, &beta, outDesc, convOutput));
}

void Conv2d::bwdFilter(cudnnHandle_t handle, void* convInput, void* dLdConv,
                       void* dLdW, void* workspace, size_t workspaceSize) {
  cudnnCheckError(cudnnConvolutionBackwardFilter(
      handle, &alpha, inDesc, convInput, outDesc, dLdConv, convDesc,
      bwdFilterAlgo, workspace, workspaceSize, &beta, filterDesc, dLdW));
}

void Conv2d::bwdData(cudnnHandle_t handle, void* convInput, void* dLdConv,
                     void* dLdIn, void* workspace, size_t workspaceSize) {
  cudnnCheckError(cudnnConvolutionBackwardData(
      handle, &alpha, filterDesc, devFilter, outDesc, dLdConv, convDesc,
      bwdDataAlgo, workspace, workspaceSize, &beta, inDesc, dLdIn));
}

size_t Conv2d::getWorkspaceSize(cudnnHandle_t handle) {
  size_t fwdSize = 0;
  cudnnCheckError(cudnnGetConvolutionForwardWorkspaceSize(
      handle, inDesc, filterDesc, convDesc, outDesc, fwdAlgo, &fwdSize));

  size_t bwdFilterSize = 0;
  cudnnCheckError(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      handle, inDesc, outDesc, convDesc, filterDesc, bwdFilterAlgo,
      &bwdFilterSize));

  size_t bwdDataSize = 0;
  cudnnCheckError(cudnnGetConvolutionBackwardDataWorkspaceSize(
      handle, filterDesc, outDesc, convDesc, inDesc, bwdDataAlgo,
      &bwdDataSize));

  return std::max({fwdSize, bwdFilterSize, bwdDataSize});
}

void Conv2d::sgd_weight(void* subtract, float learningRate) {
  float updateAlpha = -learningRate;
  cublasCheckError(cublasSaxpy_v2(cublasHandle, filterSize, &updateAlpha,
                                  (const float*)subtract, 1, (float*)devFilter,
                                  1));
}
Conv2d::~Conv2d() {
  cudaCheckError(cudaFree(devFilter));

  cudnnCheckError(cudnnDestroyConvolutionDescriptor(convDesc));
  cudnnCheckError(cudnnDestroyFilterDescriptor(filterDesc));

  cudnnCheckError(cudnnDestroyTensorDescriptor(inDesc));
  cudnnCheckError(cudnnDestroyTensorDescriptor(outDesc));
}

}  // namespace ml