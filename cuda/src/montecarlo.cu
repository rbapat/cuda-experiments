#include "montecarlo.cuh"

namespace montecarlo {

    __global__ void estimationKernel(curandState* states, int* blocksInCircle) {
        __shared__ bool in_circle[1024]; 

        ulong idx = threadIdx.x + blockIdx.x * blockDim.x;

        float x = 2 * curand_uniform(states + idx) - 1;
        float y = 2 * curand_uniform(states + idx) - 1;
        in_circle[threadIdx.x] = x*x + y*y <= 1;

        __syncthreads();

        if (threadIdx.x == 0) {
            int sum = 0;
            for (int i = 0; i < blockDim.x; i++) {
                sum += in_circle[i] ? 1 : 0;
            }

            blocksInCircle[blockIdx.x] = sum;
        }
    }

    __global__ void populateRandStates(curandState* states) {
        ulong idx = threadIdx.x + blockIdx.x * blockDim.x;
        curand_init(600382584, idx, 0, states + idx);
    }

    float estimate_pi(int numSamples, int tpb) {
        if (numSamples % tpb != 0) {
            throw std::runtime_error("numSamples must be divisible by tpb");
        }

        int numBlocks = numSamples / tpb;
        int numBytes = numBlocks * sizeof(int);
        curandState* randStates;
        cudaMalloc(&randStates, sizeof(curandState) * 1024);
        populateRandStates<<<32, 32>>>(randStates);
        cudaDeviceSynchronize();

        int* deviceBlocksInCircle = nullptr;
        cudaError_t mallocRes = cudaMalloc(&deviceBlocksInCircle, numBytes);
        cudaError_t fillRes = cudaMemset(deviceBlocksInCircle, 0, numBytes);
        if (mallocRes != cudaSuccess || fillRes != cudaSuccess) {
            throw std::runtime_error("Failed to allocate deviceBlocksInCircle buffer");
        }

        estimationKernel<<<numBlocks, tpb>>>(randStates, deviceBlocksInCircle);
        cudaDeviceSynchronize();
        
        int* hostBlocksInCircle = (int*)malloc(numBytes);
        if (hostBlocksInCircle == 0) {
            throw std::runtime_error("Failed to malloc that many bytes");
        }
        if (cudaMemcpy(hostBlocksInCircle, deviceBlocksInCircle, numBytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("%s\n", cudaGetErrorString(cudaGetLastError()));
            throw std::runtime_error("Failed to copy device to host blocks in circle");
        }

        int sum = 0.f;
        for (int i = 0; i < numBlocks; i++) {
            sum += hostBlocksInCircle[i];
        }

        return 4.f * (float)sum / numSamples;
    }


    float time_pi_estimate(int numReps, int numSamples, int tpb) {
        float avgTimeUs = 0;
        
        for (int i = 0; i < numReps; i++) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            estimate_pi(numSamples, tpb);

            cudaEventRecord(start);

            estimate_pi(numSamples, tpb);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            avgTimeUs += milliseconds * 1000 / numReps;
        }

        return avgTimeUs;
    }
}