#include "nn_softmax.h"

namespace ml {
Softmax::Softmax(cudnnHandle_t cudnnHandle, int batchSize, int numPreds)
    : cudnnHandle(cudnnHandle), batchSize(batchSize), numPreds(numPreds) {
  cudnnCheckError(cudnnCreateTensorDescriptor(&desc));

  ml::initStrides(inDimA, inStrideA, batchSize, numPreds, 1, 1);
  cudnnCheckError(
      cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, 4, inDimA, inStrideA));
}

void Softmax::fwd(void* in, void* out) {
  cudnnCheckError(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE,
                                      CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, desc,
                                      in, &beta, desc, out));
}

void Softmax::bwdData(void* in, void* out, void* dLdOut, void* dLdIn) {
  cudnnCheckError(cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE,
                                       CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, desc,
                                       out, desc, dLdOut, &beta, desc, dLdIn));
}

Softmax::~Softmax() { cudnnCheckError(cudnnDestroyTensorDescriptor(desc)); }

}  // namespace ml