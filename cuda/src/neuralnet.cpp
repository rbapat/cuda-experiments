#include "neuralnet.h"

namespace ml {
namespace fwd {
cudnn_frontend::Operation Conv2d(int64_t batchSize, int64_t inChannels,
                                 int64_t inHeight, int64_t inWidth,
                                 int64_t kernelSize, int64_t outChannels,
                                 char opName, bool virtualIn, bool virtualOut,
                                 int64_t inputId, int64_t outputId) {
  constexpr int64_t numDims = 2;
  int64_t weightId = (opName << 16) + (CONV_OP_ID << 8) + 'w';

  auto inputTensor = ml::createTensor(inputId, virtualIn, batchSize, inChannels,
                                      inWidth, inHeight);

  auto outputTensor = ml::createTensor(outputId, virtualOut, batchSize,
                                       outChannels, inWidth, inHeight);

  auto weightTensor = ml::createTensor(weightId, true, outChannels, batchSize,
                                       kernelSize, kernelSize);

  int64_t convStride[numDims], convPadding[numDims], convDilation[numDims];
  for (size_t i = 0; i < numDims; i++) {
    convStride[i] = convDilation[i] = 1;
    convPadding[i] = kernelSize / 2;
  }

  auto convDesc = cudnn_frontend::ConvDescBuilder()
                      .setComputeType(CUDNN_DATA_FLOAT)
                      .setMathMode(CUDNN_CONVOLUTION)
                      .setSpatialDimCount(numDims)
                      .setSpatialStride(numDims, convStride)
                      .setPrePadding(numDims, convPadding)
                      .setPostPadding(numDims, convPadding)
                      .setDilation(numDims, convDilation)
                      .build();

  return cudnn_frontend::OperationBuilder(
             CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
      .setxDesc(inputTensor)
      .setyDesc(outputTensor)
      .setwDesc(weightTensor)
      .setcDesc(convDesc)
      .setAlpha(1.0f)
      .setBeta(0.0f)
      .build();
}

cudnn_frontend::Operation Bias(int64_t batchSize, int64_t numChannels,
                               int64_t height, int64_t width, char opName,
                               bool virtualIn, bool virtualOut, int64_t inputId,
                               int64_t outputId) {
  auto addDesc = cudnn_frontend::PointWiseDescBuilder()
                     .setMode(CUDNN_POINTWISE_ADD)
                     .setMathPrecision(CUDNN_DATA_FLOAT)
                     .build();

  int64_t biasId = (opName << 16) + (BIAS_OP_ID << 8) + 'b';

  auto inputTensor = ml::createTensor(inputId, virtualIn, batchSize,
                                      numChannels, width, height);

  auto outputTensor = ml::createTensor(outputId, virtualOut, batchSize,
                                       numChannels, width, height);

  auto biasTensor = ml::createTensor(biasId, true, 1, numChannels, 1, 1);

  return cudnn_frontend::OperationBuilder(
             CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(inputTensor)
      .setbDesc(biasTensor)
      .setyDesc(outputTensor)
      .setpwDesc(addDesc)
      .setAlpha(1.0)
      .setAlpha(1.0)
      .build();
}

cudnn_frontend::Operation ReLU(int64_t batchSize, int64_t numChannels,
                               int64_t height, int64_t width, char opName,
                               bool virtualIn, bool virtualOut, int64_t inputId,
                               int64_t outputId) {
  auto reluDesc = cudnn_frontend::PointWiseDescBuilder()
                      .setMode(CUDNN_POINTWISE_RELU_FWD)
                      .setMathPrecision(CUDNN_DATA_FLOAT)
                      .build();

  auto inputTensor = ml::createTensor(inputId, virtualIn, batchSize,
                                      numChannels, width, height);

  auto outputTensor = ml::createTensor(outputId, virtualOut, batchSize,
                                       numChannels, width, height);

  return cudnn_frontend::OperationBuilder(
             CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(inputTensor)
      .setyDesc(outputTensor)
      .setpwDesc(reluDesc)
      .setAlpha(1.0)
      .build();
}

/*
  int64_t convStride[numDims], convPadding[numDims], convDilation[numDims];
  for (size_t i = 0; i < numDims; i++) {
    convStride[i] = convDilation[i] = 1;
    convPadding[i] = kernelSize / 2;
  }

  auto convDesc = cudnn_frontend::ConvDescBuilder()
                      .setComputeType(CUDNN_DATA_FLOAT)
                      .setMathMode(CUDNN_CONVOLUTION)
                      .setSpatialDimCount(numDims)
                      .setSpatialStride(numDims, convStride)
                      .setPrePadding(numDims, convPadding)
                      .setPostPadding(numDims, convPadding)
                      .setDilation(numDims, convDilation)
                      .build();
*/
cudnn_frontend::Operation Pool(int64_t batchSize, int64_t numChannels,
                               int64_t height, int64_t width,
                               int64_t kernelSize, int64_t stride,
                               cudnnResampleMode_t poolType, char opName,
                               bool virtualIn, bool virtualOut, int64_t inputId,
                               int64_t outputId) {
  constexpr size_t numDims = 2;
  int64_t kernelDim[numDims], poolStride[numDims], poolPadding[numDims];
  for (size_t i = 0; i < numDims; i++) {
    // We will assume height=width, and it is divisible by kernelSize and stride
    kernelDim[i] = kernelSize;
    poolStride[i] = stride;
    poolPadding[i] = 0;  // kernelSize / 2;
  }

  auto poolDesc = cudnn_frontend::ResampleDescBuilder()
                      .setComputeType(CUDNN_DATA_FLOAT)
                      .setNanPropagation(CUDNN_NOT_PROPAGATE_NAN)
                      .setPaddingMode(CUDNN_NEG_INF_PAD)
                      .setPostPadding(numDims, poolPadding)
                      .setPrePadding(numDims, poolPadding)
                      .setResampleMode(poolType)
                      .setSpatialDim(numDims, kernelDim)
                      .setSpatialStride(numDims, poolStride)
                      .build();

  auto inputTensor = ml::createTensor(inputId, virtualIn, batchSize,
                                      numChannels, width, height);

  auto outputTensor =
      ml::createTensor(outputId, virtualOut, batchSize, numChannels,
                       width / stride, height / stride);

  return cudnn_frontend::OperationBuilder(
             CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
      .setxDesc(inputTensor)
      .setyDesc(outputTensor)
      .setResampleDesc(poolDesc)
      .build();
}
}  // namespace fwd
}  // namespace ml