#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <numeric>

#include "benchmark.h"
#include "cudacommon.h"
#include "simplecnn.cuh"
#include "thirdparty/mnist/mnist_reader.hpp"

#define MNIST_DATA_LOCATION "/home/rohan/projects/cuda-experiments/cuda/assets"

namespace mnist {

class CNN : public benchmark::TimedAlgorithm {
 public:
  CNN(int batch_size);
  void calculate();
  std::string_view getName();
  ~CNN();

 private:
  void writeDatasetToDevice();
  void populateDeviceData(int numSamples,
                          std::vector<std::vector<uint8_t>> hostImages,
                          std::vector<uint8_t> hostLabels,
                          uint8_t** deviceImages, uint8_t** deviceLabels);

  uint8_t* devTrainImages;
  uint8_t* devTrainLabels;
  uint8_t* devTestImages;
  uint8_t* devTestLabels;

  ml::SimpleCNN* cnn;

  int numTrain, numTest, batchSize;
};

}  // namespace mnist