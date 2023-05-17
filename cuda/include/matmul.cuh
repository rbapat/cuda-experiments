#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "benchmark.h"
#include "matrix.h"

namespace matmul {

class Naive : public benchmark::TimedAlgorithm {
 public:
  Naive(int _matrixDim, int _threadsPerBlock);
  void calculate();
  std::string_view getName();
  ~Naive();

 private:
  int matrixDim;
  int threadsPerBlock;

  float* mat1;
  float* mat2;
  float* out;
};

class Cublas : public benchmark::TimedAlgorithm {
 public:
  Cublas(int _matrixDim);
  std::string_view getName();
  void calculate();
  ~Cublas();

 private:
  int matrixDim;

  float* mat1;
  float* mat2;
  float* out;

  cublasHandle_t handle;
  const float alpha, beta;
};
}  // namespace matmul