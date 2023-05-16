#include <cub/cub.cuh>

#include "simplecnn.cuh"

namespace ml {

SimpleCNN::SimpleCNN(int batchSize) : batchSize(batchSize) {
  cudnnCheckError(cudnnCreate(&cudnnHandle));
  cublasCheckError(cublasCreate(&cublasHandle));

  cudnnCheckError(cudnnCreateTensorDescriptor(&desc1));
  cudnnCheckError(cudnnCreateTensorDescriptor(&desc2));
  cudnnCheckError(cudnnCreateTensorDescriptor(&desc3));
  cudnnCheckError(cudnnCreateTensorDescriptor(&desc4));

  conv1 = new Conv2d(cublasHandle, batchSize, 28, 28, 32, 1, 3, 1, 1);
  bias1 = new Bias(cublasHandle, 32);
  relu1 = new ReLU();
  pool1 = new Pool2d(batchSize, 32, 28, 28, 2, 0, 2);

  conv2 = new Conv2d(cublasHandle, batchSize, 14, 14, 64, 32, 3, 1, 1);
  bias2 = new Bias(cublasHandle, 64);
  relu2 = new ReLU();
  pool2 = new Pool2d(batchSize, 64, 14, 14, 2, 0, 2);

  linear1 = new Linear(cublasHandle, batchSize, 64 * 7 * 7, 256);
  bias3 = new Bias(cublasHandle, 256);
  relu3 = new ReLU();
  linear2 = new Linear(cublasHandle, batchSize, 256, 10);
  bias4 = new Bias(cublasHandle, 10);

  softmax = new Softmax(cudnnHandle, batchSize, 10);

  workspaceSize = std::max({conv1->getWorkspaceSize(cudnnHandle),
                            conv2->getWorkspaceSize(cudnnHandle)});
  cudaCheckError(cudaMalloc(&workspace, workspaceSize));

  cudaCheckError(
      cudaMalloc(&conv1Out, batchSize * 32 * 28 * 28 * sizeof(float)));
  cudaCheckError(
      cudaMalloc(&relu1Out, batchSize * 32 * 28 * 28 * sizeof(float)));
  cudaCheckError(
      cudaMalloc(&pool1Out, batchSize * 32 * 14 * 14 * sizeof(float)));
  cudaCheckError(
      cudaMalloc(&conv2Out, batchSize * 64 * 14 * 14 * sizeof(float)));
  cudaCheckError(
      cudaMalloc(&relu2Out, batchSize * 64 * 14 * 14 * sizeof(float)));
  cudaCheckError(cudaMalloc(&pool2Out, batchSize * 64 * 7 * 7 * sizeof(float)));
  cudaCheckError(cudaMalloc(&linear1Out, batchSize * 256 * sizeof(float)));
  cudaCheckError(cudaMalloc(&relu3Out, batchSize * 256 * sizeof(float)));
  cudaCheckError(cudaMalloc(&linear2Out, batchSize * 10 * sizeof(float)));
  cudaCheckError(cudaMalloc(&softmaxOut, batchSize * 10 * sizeof(float)));

  // allocate space for tracking gradients
  cudaCheckError(cudaMalloc(&dLdConv1, batchSize * 32 * 3 * 3 * sizeof(float)));
  cudaCheckError(cudaMalloc(&dLdBias1, batchSize * 32 * sizeof(float)));
  cudaCheckError(cudaMalloc(&dLdConv2, batchSize * 64 * 3 * 3 * sizeof(float)));
  cudaCheckError(cudaMalloc(&dLdBias2, batchSize * 64 * sizeof(float)));
  cudaCheckError(
      cudaMalloc(&dLdLinear1W, batchSize * 64 * 7 * 7 * 256 * sizeof(float)));
  cudaCheckError(cudaMalloc(&dLdLinear1B, batchSize * 256 * sizeof(float)));
  cudaCheckError(
      cudaMalloc(&dLdLinear2W, batchSize * 256 * 10 * sizeof(float)));
  cudaCheckError(cudaMalloc(&dLdLinear2B, batchSize * 10 * sizeof(float)));
  cudaCheckError(cudaMalloc(&dLoss, batchSize * 10 * sizeof(float)));
  cudaCheckError(cudaMalloc(&loss, batchSize * sizeof(float)));

  cudaCheckError(cudaMalloc(&dLdInput, batchSize * 28 * 28 * sizeof(float)));
  cudaCheckError(
      cudaMalloc(&dLdConv1Out, batchSize * 32 * 28 * 28 * sizeof(float)));
  cudaCheckError(
      cudaMalloc(&dLdRelu1Out, batchSize * 32 * 28 * 28 * sizeof(float)));
  cudaCheckError(
      cudaMalloc(&dLdPool1Out, batchSize * 32 * 14 * 14 * sizeof(float)));
  cudaCheckError(
      cudaMalloc(&dLdConv2Out, batchSize * 64 * 14 * 14 * sizeof(float)));
  cudaCheckError(cudaMalloc(&dLdRelu3Out, batchSize * 256 * sizeof(float)));
  cudaCheckError(
      cudaMalloc(&dLdRelu2Out, batchSize * 64 * 14 * 14 * sizeof(float)));
  cudaCheckError(
      cudaMalloc(&dLdPool2Out, batchSize * 64 * 7 * 7 * sizeof(float)));
  cudaCheckError(cudaMalloc(&dLdLin1Out, batchSize * 256 * sizeof(float)));
  cudaCheckError(cudaMalloc(&dLdLinear2Out, batchSize * 10 * sizeof(float)));

  // set misc descriptors
  int dims1[4] = {batchSize, 32, 28, 28};
  int strides1[4] = {25088, 784, 28, 1};
  cudnnCheckError(
      cudnnSetTensorNdDescriptor(desc1, CUDNN_DATA_FLOAT, 4, dims1, strides1));

  int dims2[4] = {batchSize, 64, 14, 14};
  int strides2[4] = {14 * 14 * 64, 14 * 14, 14, 1};
  cudnnCheckError(
      cudnnSetTensorNdDescriptor(desc2, CUDNN_DATA_FLOAT, 4, dims2, strides2));

  int dims3[4] = {batchSize, 256, 1, 1};
  int strides3[4] = {256, 1, 1, 1};
  cudnnCheckError(
      cudnnSetTensorNdDescriptor(desc3, CUDNN_DATA_FLOAT, 4, dims3, strides3));

  int dims4[4] = {batchSize, 10, 1, 1};
  int strides4[4] = {10, 1, 1, 1};
  cudnnCheckError(
      cudnnSetTensorNdDescriptor(desc4, CUDNN_DATA_FLOAT, 4, dims4, strides4));

  cudaCheckError(cudaMalloc(&sumLoss, sizeof(float)));
  CubDebugExit(cub::DeviceReduce::Sum(tempStorage, tempStorageSize,
                                      (float*)loss, sumLoss, batchSize));

  cudaCheckError(cudaMalloc(&tempStorage, tempStorageSize));
  cudaCheckError(cudaDeviceSynchronize());
}

__global__ void crossEntropyLoss(float* softmaxProbs, uint8_t* labels,
                                 size_t batchSize, size_t numClasses,
                                 float* losses, float* gradients) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batchIdx = idx * numClasses;

  while (idx < batchSize) {
    uint8_t label = labels[idx];
    losses[batchIdx] = -logf(softmaxProbs[batchIdx + label]);

    for (size_t i = 0; i < numClasses; i++) {
      gradients[batchIdx + i] =
          softmaxProbs[batchIdx + i] - (i == label ? 1.0f : 0.0f);
    }

    idx += gridDim.x * blockDim.x;
  }
}

float SimpleCNN::run(void* input, uint8_t* labels, float learningRate) {
  // forward propagation
  conv1->fwd(cudnnHandle, input, conv1Out, workspace, workspaceSize);
  bias1->fwd(cudnnHandle, desc1, conv1Out);
  relu1->fwd(cudnnHandle, desc1, conv1Out, relu1Out);
  pool1->fwd(cudnnHandle, relu1Out, pool1Out);
  conv2->fwd(cudnnHandle, pool1Out, conv2Out, workspace, workspaceSize);
  bias2->fwd(cudnnHandle, desc2, conv2Out);
  relu2->fwd(cudnnHandle, desc2, conv2Out, relu2Out);
  pool2->fwd(cudnnHandle, relu2Out, pool2Out);
  linear1->fwd(pool2Out, linear1Out);
  bias3->fwd(cudnnHandle, desc3, linear1Out);
  relu3->fwd(cudnnHandle, desc3, linear1Out, relu3Out);
  linear2->fwd(relu3Out, linear2Out);
  bias4->fwd(cudnnHandle, desc4, linear2Out);
  softmax->fwd(linear2Out, softmaxOut);

  dim3 gridDim((batchSize / 128) + 1);
  dim3 blockDim(128);
  crossEntropyLoss<<<gridDim, blockDim>>>((float*)softmaxOut, labels, batchSize,
                                          10, (float*)loss, (float*)dLoss);

  // back propagation
  softmax->bwdData(linear2Out, softmaxOut, dLoss, dLdLinear2Out);
  linear2->bwdW(relu3Out, dLdLinear2Out, dLdLinear2W);
  linear2->bwdData(dLdLinear2Out, dLdRelu3Out);

  relu3->bwdData(cudnnHandle, desc3, linear1Out, relu3Out, dLdRelu3Out,
                 dLdLin1Out);

  linear1->bwdW(pool2Out, dLdLin1Out, dLdLinear1W);
  linear1->bwdData(dLdLin1Out, dLdPool2Out);

  pool2->bwdData(cudnnHandle, relu2Out, pool2Out, dLdPool2Out, dLdRelu2Out);
  relu2->bwdData(cudnnHandle, desc2, conv2Out, relu2Out, dLdRelu2Out,
                 dLdConv2Out);
  conv2->bwdFilter(cudnnHandle, pool1Out, dLdConv2Out, dLdConv2, workspace,
                   workspaceSize);
  conv2->bwdData(cudnnHandle, pool1Out, dLdConv2Out, dLdPool1Out, workspace,
                 workspaceSize);
  pool1->bwdData(cudnnHandle, relu1Out, pool1Out, dLdPool1Out, dLdRelu1Out);
  relu1->bwdData(cudnnHandle, desc1, conv1Out, relu1Out, dLdRelu1Out,
                 dLdConv1Out);
  conv1->bwdFilter(cudnnHandle, input, dLdConv1Out, dLdConv1, workspace,
                   workspaceSize);

  cudaCheckError(cudaMalloc(&tempStorage, tempStorageSize));
  conv1->bwdData(cudnnHandle, input, dLdConv1Out, dLdInput, workspace,
                 workspaceSize);

  // gradient update
  bias4->sgd_bias(dLoss, learningRate);
  linear2->sgd_weight(dLdLinear2W, learningRate);
  bias3->sgd_bias(dLdLin1Out, learningRate);
  linear1->sgd_weight(dLdLinear1W, learningRate);
  bias2->sgd_bias(dLdConv2Out, learningRate);
  conv2->sgd_weight(dLdConv2, learningRate);
  bias1->sgd_bias(dLdConv1Out, learningRate);
  conv1->sgd_weight(dLdConv1, learningRate);

  CubDebugExit(cub::DeviceReduce::Sum(tempStorage, tempStorageSize,
                                      (float*)loss, sumLoss, batchSize));

  cudaCheckError(cudaDeviceSynchronize());
  float hostOut = 0;
  cudaCheckError(
      cudaMemcpy(&hostOut, sumLoss, sizeof(float), cudaMemcpyDeviceToHost));

  return hostOut;
}
SimpleCNN::~SimpleCNN() {
  cudnnDestroyTensorDescriptor(desc1);
  cudnnDestroyTensorDescriptor(desc2);
  cudnnDestroy(cudnnHandle);
  cublasDestroy(cublasHandle);

  delete conv1;
  delete bias1;
  delete relu1;
  delete pool1;
  delete conv2;
  delete bias2;
  delete relu2;
  delete pool2;
  delete linear1;
  delete linear2;

  // i should probably free all that gpu memory
}
}  // namespace ml