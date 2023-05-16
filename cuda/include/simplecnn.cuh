#pragma once
#include <cudnn_frontend.h>

#include "nn_bias.h"
#include "nn_conv.h"
#include "nn_linear.h"
#include "nn_pool.h"
#include "nn_relu.h"
#include "nn_softmax.h"

namespace ml {

class SimpleCNN {
 public:
  SimpleCNN(int batchSize);

  float run(void* input, uint8_t* labels, float learningRate);

  ~SimpleCNN();

 private:
  cudnnHandle_t cudnnHandle;
  cublasHandle_t cublasHandle;

  void* workspace;
  int workspaceSize, batchSize;

  Conv2d* conv1;
  Bias* bias1;
  ReLU* relu1;
  Pool2d* pool1;
  Conv2d* conv2;
  Bias* bias2;
  ReLU* relu2;
  Pool2d* pool2;
  Linear* linear1;
  Bias* bias3;
  ReLU* relu3;
  Linear* linear2;
  Bias* bias4;
  Softmax* softmax;

  cudnnTensorDescriptor_t desc1;
  cudnnTensorDescriptor_t desc2;
  cudnnTensorDescriptor_t desc3;
  cudnnTensorDescriptor_t desc4;

  // intermediate tensors for computing gradients
  // void* input;
  void* conv1Out;
  void* relu1Out;
  void* pool1Out;
  void* conv2Out;
  void* relu2Out;
  void* pool2Out;
  void* linear1Out;
  void* relu3Out;
  void* linear2Out;
  void* softmaxOut;

  void* dLdInput;
  void* dLdConv1Out;
  void* dLdRelu1Out;
  void* dLdLin1Out;
  void* dLdPool2Out;
  void* dLdRelu2Out;
  void* dLdConv2Out;
  void* dLdPool1Out;
  void* dLdRelu3Out;
  void* dLdLinear2Out;

  // gradient tensors
  void* dLdConv1;
  void* dLdBias1;
  void* dLdConv2;
  void* dLdBias2;
  void* dLdLinear1W;
  void* dLdLinear1B;
  void* dLdLinear2W;
  void* dLdLinear2B;

  void* loss;
  void* dLoss;
  float* sumLoss;

  float* tempStorage;
  size_t tempStorageSize = 0;
};

}  // namespace ml