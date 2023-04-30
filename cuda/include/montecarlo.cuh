#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>

#include "benchmark.h"
namespace montecarlo {
class Naive : public benchmark::TimedAlgorithm {
 public:
  Naive(int _numSamples, int _blocksPerGrid, int _threadsPerBlock);
  std::string_view getName();
  void calculate();
  ~Naive();

 private:
  float naiveReduce(int* simSums);
  float cubReduce(int* simSums);

  int numSamples;
  int blocksPerGrid;
  int threadsPerBlock;
  curandState* randStates;
};
}  // namespace montecarlo