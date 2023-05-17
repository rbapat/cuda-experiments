#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "benchmark.h"
#include "cudacommon.h"

#define IRIS_DATA_FILE \
  "/home/rohan/projects/cuda-experiments/data/iris/iris.data"

#define LEARNING_RATE 1.0f
namespace iris {

class LogisticRegression : public benchmark::TimedAlgorithm {
 public:
  LogisticRegression(int numEpoch);
  void calculate();
  std::string_view getName();
  ~LogisticRegression();

 private:
  int numEpoch, numSamples;

  cublasHandle_t handle;
  int loadData(void** devDataPtr, void** devLabelPtr);
  void* devDataPtr;
  void* devLabelPtr;
};
}  // namespace iris