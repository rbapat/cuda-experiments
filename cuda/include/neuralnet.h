#pragma once

#include <cudnn_frontend.h>

#include "benchmark.h"

#define CONV_OP_ID 'C'
#define BIAS_OP_ID 'B'
#define RELU_OP_ID 'R'
#define POOL_OP_ID 'P'

namespace ml {

template <typename... Args>
cudnn_frontend::Tensor createTensor(int64_t id, bool isVirtual, Args... args) {
  const size_t numDims = sizeof...(args);

  size_t index = 0;
  int64_t dims[numDims];
  ((dims[index++] = static_cast<int>(args)), ...);

  int64_t strides[numDims] = {0};
  strides[numDims - 1] = 1;
  for (int i = numDims - 2; i >= 0; i--)
    strides[i] = dims[i + 1] * strides[i + 1];

  auto& builder = cudnn_frontend::TensorBuilder()
                      .setDim(numDims, dims)
                      .setDataType(CUDNN_DATA_FLOAT)
                      .setAlignment(numDims)
                      .setId(id)
                      .setStride(numDims, strides);

  return (isVirtual ? builder.setVirtual() : builder).build();
}

namespace fwd {
cudnn_frontend::Operation Conv2d(int64_t batchSize, int64_t inChannels,
                                 int64_t inHeight, int64_t inWidth,
                                 int64_t kernelSize, int64_t outChannels,
                                 char opName, bool virtualIn, bool virtualOut,
                                 int64_t inputId, int64_t outputId);

cudnn_frontend::Operation Bias(int64_t batchSize, int64_t numChannels,
                               int64_t height, int64_t width, char opName,
                               bool virtualIn, bool virtualOut, int64_t inputId,
                               int64_t outputId);

cudnn_frontend::Operation ReLU(int64_t batchSize, int64_t numChannels,
                               int64_t height, int64_t width, char opName,
                               bool virtualIn, bool virtualOut, int64_t inputId,
                               int64_t outputId);

cudnn_frontend::Operation Pool(int64_t batchSize, int64_t numChannels,
                               int64_t height, int64_t width,
                               int64_t kernelSize, int64_t stride,
                               cudnnResampleMode_t poolType, char opName,
                               bool virtualIn, bool virtualOut, int64_t inputId,
                               int64_t outputId);
}  // namespace fwd
}  // namespace ml