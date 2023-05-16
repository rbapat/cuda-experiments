#pragma once
#include <cudnn_frontend.h>

#include "benchmark.h"
#include "neuralnet.h"

#define cudnnCheckError(err)                                   \
  {                                                            \
    if (err != CUDNN_STATUS_SUCCESS) {                         \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudnnGetErrorString(err));                        \
      exit(0);                                                 \
    }                                                          \
  }

namespace mnist {

class DNN : public benchmark::TimedAlgorithm {
 public:
  DNN();
  void calculate();
  std::string_view getName();
  ~DNN();

 private:
  cudnnHandle_t handle;
  std::shared_ptr<cudnn_frontend::ExecutionPlan> execPlan;

  void* devInput;
  void* devConvWeight1;
  void* devBias1;
  void* devConvWeight2;
  void* devBias2;
  void* devPoolOut;
};

}  // namespace mnist