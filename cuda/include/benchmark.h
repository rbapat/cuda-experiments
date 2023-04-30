#pragma once
#include <cuda_runtime.h>

#include <string>

#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(0);                                                 \
    }                                                          \
  }

namespace benchmark {
class TimedAlgorithm {
 public:
  TimedAlgorithm() {}
  virtual void calculate() = 0;
  virtual std::string_view getName() = 0;
  virtual ~TimedAlgorithm() {}
};

float time_algorithm(TimedAlgorithm* algo, int numReps);
}  // namespace benchmark