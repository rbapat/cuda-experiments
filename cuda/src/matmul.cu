#include "matmul.cuh"

namespace matmul {

__global__ void matmul_kernel(float* mat1, float* mat2, float* out,
                              int matrixDim) {
  int idxX = blockIdx.x * blockDim.x + threadIdx.x;
  int idxY = blockIdx.y * blockDim.y + threadIdx.y;

  float total = 0;
  for (int i = 0; i < matrixDim; i++) {
    total += mat1[matrixDim * idxY + i] * mat2[matrixDim * idxX + i];
  }

  out[idxY * matrixDim + idxX] = total;
}

Naive::Naive(int _matrixDim, int _threadsPerBlock)
    : matrixDim(_matrixDim), threadsPerBlock(_threadsPerBlock) {
  if (matrixDim % threadsPerBlock != 0) {
    throw std::invalid_argument(
        "matrixDim must be a multiple of threadsPerBlock");
  }

  mat1 = matrix::generateRandomMatrixCuda(matrixDim);
  mat2 = matrix::generateRandomMatrixCuda(matrixDim);
  out = matrix::allocateEmptyMatrixCuda(matrixDim);
}

void Naive::calculate() {
  dim3 gridDim(matrixDim / threadsPerBlock, matrixDim / threadsPerBlock);
  dim3 blockDim(threadsPerBlock, threadsPerBlock);

  matmul_kernel<<<gridDim, blockDim>>>(mat1, mat2, out, matrixDim);
}

std::string_view Naive::getName() { return "Naive Matmul"; }

Naive::~Naive() {
  cudaFree(mat1);
  cudaFree(mat2);
  cudaFree(out);
}

Cublas::Cublas(int _matrixDim)
    : matrixDim(_matrixDim), alpha(1.0f), beta(0.0f) {
  mat1 = matrix::generateRandomMatrixCuda(matrixDim);
  mat2 = matrix::generateRandomMatrixCuda(matrixDim);
  out = matrix::allocateEmptyMatrixCuda(matrixDim);

  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create cublas handle");
  }
}

std::string_view Cublas::getName() { return "cuBLAS Matmul"; }

void Cublas::calculate() {
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrixDim, matrixDim, matrixDim,
              &alpha, mat2, matrixDim, mat1, matrixDim, &beta, out, matrixDim);
}

Cublas::~Cublas() {
  cublasDestroy(handle);

  cudaFree(mat1);
  cudaFree(mat2);
  cudaFree(out);
}

}  // namespace matmul