#include <iostream>
#include <stdlib.h>

float* allocateEmptyMatrixCuda(const int len) {
    float* ptr = nullptr;
    cudaError_t res = cudaMalloc(&ptr, len * len * sizeof(float));

    if (res != cudaSuccess) {
        throw std::runtime_error("Unable to allocate memory for empty cuda matrix");
    }

    return ptr;    
}

float* allocateEmptyMatrixCpu(const int len) {
    return (float*)calloc(len * len, sizeof(float));
}

/*
Let's assume matrices here are row-major, so the matrix:

A B C
D E F
G H I

is represented as `A B C D E F G H I` in memory
*/
float* generateRandomMatrixCuda(const int len) {
    float* cpuMat = allocateEmptyMatrixCpu(len);
    for (int i = 0; i < len * len; i++) {
        cpuMat[i] = (float)rand() / RAND_MAX;
    }

    float* gpuMat = allocateEmptyMatrixCuda(len);
    cudaError_t res = cudaMemcpy(gpuMat, cpuMat, len * len * sizeof(float), cudaMemcpyHostToDevice);

    free(cpuMat);
    if (res != cudaSuccess) {
        throw std::runtime_error("Failed to copy randomly allocated matrix from host to device");
    }

    return gpuMat;
}

__global__ void matmul(float* mat1, float* mat2, float *out, int matrixSize) {
    int idxX = blockIdx.x * blockDim.x + threadIdx.x;
    int idxY = blockIdx.y * blockDim.y + threadIdx.y;

    float total = 0;
    for (int i = 0; i < matrixSize; i++) {
        total += mat1[matrixSize * idxY + i] * mat2[matrixSize * idxX + i];
    }

    out[idxY * matrixSize + idxX] = total;
}

int main() {
    constexpr int matrixSize = 512;
    constexpr int threadsPerBlock = 16;
    if (matrixSize % threadsPerBlock != 0) {
        throw std::runtime_error("matrixSize must be a multiple of threadsPerBlock");
    }

    float* mat1 = generateRandomMatrixCuda(matrixSize);
    float* mat2 = generateRandomMatrixCuda(matrixSize);
    float* out = allocateEmptyMatrixCuda(matrixSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 gridDim(matrixSize / threadsPerBlock, matrixSize / threadsPerBlock);
    dim3 blockDim(threadsPerBlock, threadsPerBlock);

    cudaEventRecord(start);
    matmul<<<gridDim, blockDim>>>(mat1, mat2, out, matrixSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaFree(mat1);
    cudaFree(mat2);
    cudaFree(out);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("matmul took %f us\n", milliseconds * 1000);
}