#include "nn_relu.h"

namespace ml {
ReLU::ReLU() {
  cudnnCheckError(cudnnCreateActivationDescriptor(&reluDesc));
  cudnnSetActivationDescriptor(reluDesc, CUDNN_ACTIVATION_RELU,
                               CUDNN_PROPAGATE_NAN, 0.0);
}

void ReLU::fwd(cudnnHandle_t handle, cudnnTensorDescriptor_t tensorDesc,
               void* in, void* out) {
  cudnnCheckError(cudnnActivationForward(handle, reluDesc, &alpha, tensorDesc,
                                         in, &beta, tensorDesc, out));
}
void ReLU::bwdData(cudnnHandle_t handle, cudnnTensorDescriptor_t tensorDesc,
                   void* in, void* out, void* dLdRelu, void* dLdIn) {
  cudnnCheckError(cudnnActivationBackward(handle, reluDesc, &alpha, tensorDesc,
                                          out, tensorDesc, dLdRelu, tensorDesc,
                                          in, &beta, tensorDesc, dLdIn));
}

ReLU::~ReLU() { cudnnCheckError(cudnnDestroyActivationDescriptor(reluDesc)); }
}  // namespace ml
