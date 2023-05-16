#pragma once
#include <cuda_runtime.h>
#include <cudnn.h>

#include <string>

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