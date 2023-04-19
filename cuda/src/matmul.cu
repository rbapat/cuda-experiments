#include "matmul.cuh"

namespace matmul {

    namespace naive {
        __global__ void matmul_kernel(float* mat1, float* mat2, float *out, int matrixSize) {
            int idxX = blockIdx.x * blockDim.x + threadIdx.x;
            int idxY = blockIdx.y * blockDim.y + threadIdx.y;

            float total = 0;
            for (int i = 0; i < matrixSize; i++) {
                total += mat1[matrixSize * idxY + i] * mat2[matrixSize * idxX + i];
            }

            out[idxY * matrixSize + idxX] = total;
        }

        float time_execution(int numTimes, int matrixSize, int tpb) {
            if (matrixSize % tpb != 0) {
                throw std::runtime_error("matrixSize must be a multiple of threadsPerBlock");
            }

            float* mat1 = matrix::generateRandomMatrixCuda(matrixSize);
            float* mat2 = matrix::generateRandomMatrixCuda(matrixSize);
            float* out = matrix::allocateEmptyMatrixCuda(matrixSize);

            float avgTimeUs = 0;

            for (int i = 0; i < numTimes; i++) {
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                
                dim3 gridDim(matrixSize / tpb, matrixSize / tpb);
                dim3 blockDim(tpb, tpb);

                cudaEventRecord(start);

                matmul_kernel<<<gridDim, blockDim>>>(mat1, mat2, out, matrixSize);

                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                avgTimeUs += milliseconds * 1000 / numTimes;
            }

            cudaFree(mat1);
            cudaFree(mat2);
            cudaFree(out);

            return avgTimeUs;
        }
    }

    namespace cublas {
        float time_execution(int numTimes, int matrixSize, int tpb) {
            if (matrixSize % tpb != 0) {
                throw std::runtime_error("matrixSize must be a multiple of threadsPerBlock");
            }

            float* mat1 = matrix::generateRandomMatrixCuda(matrixSize);
            float* mat2 = matrix::generateRandomMatrixCuda(matrixSize);
            float* out = matrix::allocateEmptyMatrixCuda(matrixSize);

            const float alpha = 1.0f;
            const float beta  = 0.0f;

            float avgTimeUs = 0;
            

            cublasHandle_t handle;
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error("Failed to create cublas handle");
            }

            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrixSize, matrixSize, matrixSize, &alpha, mat2, matrixSize, mat1, matrixSize, &beta, out, matrixSize);

            for (int i = 0; i < numTimes; i++) {
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                
                dim3 gridDim(matrixSize / tpb, matrixSize / tpb);
                dim3 blockDim(tpb, tpb);

                cudaEventRecord(start);

                cublasStatus_t res = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrixSize, matrixSize, matrixSize, &alpha, mat2, matrixSize, mat1, matrixSize, &beta, out, matrixSize);

                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                if (res != CUBLAS_STATUS_SUCCESS) {
                    std::cout << "Failed to do cublasSgemm at iteration " << i << " with error " << cudaGetErrorString(cudaGetLastError()) << std::endl;
                }
                
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                avgTimeUs += milliseconds * 1000 / numTimes;
            }

            cudaFree(mat1);
            cudaFree(mat2);
            cudaFree(out);

            return avgTimeUs;
        }
    }
}