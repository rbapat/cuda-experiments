#include <cub/cub.cuh>

#include "montecarlo.cuh"
namespace montecarlo {

__global__ void populateRandStates(curandState* randStates) {
  ulong idx = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(600382584, idx, 0, randStates + idx);
}

__global__ void estimate_pi(curandState* randStates, int* out_arr,
                            int numReps) {
  ulong idx = threadIdx.x + blockIdx.x * blockDim.x;

  int inCircle = 0;
  for (size_t rep = 0; rep < numReps; rep++) {
    float x = 2 * curand_uniform(randStates + idx) - 1;
    float y = 2 * curand_uniform(randStates + idx) - 1;

    if (x * x + y * y <= 1) inCircle++;
  }

  out_arr[idx] = inCircle;
}

Naive::Naive(int _numSamples, int _blocksPerGrid, int _threadsPerBlock)
    : numSamples(_numSamples),
      blocksPerGrid(_blocksPerGrid),
      threadsPerBlock(_threadsPerBlock) {
  cudaCheckError(cudaMalloc(
      &randStates, sizeof(curandState) * blocksPerGrid * threadsPerBlock));

  const dim3 gridSize(blocksPerGrid);
  const dim3 blockSize(threadsPerBlock);

  populateRandStates<<<gridSize, blockSize>>>(randStates);
}

std::string_view Naive::getName() { return "Monte Carlo Pi Estimation"; }

float Naive::naiveReduce(int* simSums) {
  int* simSumsHost =
      (int*)malloc(blocksPerGrid * threadsPerBlock * sizeof(int));
  cudaCheckError(cudaMemcpy(simSumsHost, simSums,
                            blocksPerGrid * threadsPerBlock * sizeof(int),
                            cudaMemcpyDeviceToHost));

  int sum = 0;
  for (size_t i = 0; i < blocksPerGrid * threadsPerBlock; i++) {
    sum += simSumsHost[i];
  }

  free(simSumsHost);
  return 4.f * sum / numSamples;
}

float Naive::cubReduce(int* simSums) {
  int* deviceOut;
  cudaCheckError(cudaMalloc(&deviceOut, sizeof(int)));

  void* tempStorage = NULL;
  size_t tempStorageSize = 0;
  const int numItems = blocksPerGrid * threadsPerBlock;

  CubDebugExit(cub::DeviceReduce::Sum(tempStorage, tempStorageSize, simSums,
                                      deviceOut, numItems));

  cudaCheckError(cudaMalloc(&tempStorage, tempStorageSize));

  CubDebugExit(cub::DeviceReduce::Sum(tempStorage, tempStorageSize, simSums,
                                      deviceOut, numItems));

  int hostOut = 0;
  cudaCheckError(
      cudaMemcpy(&hostOut, deviceOut, sizeof(int), cudaMemcpyDeviceToHost));

  cudaFree(deviceOut);
  cudaFree(tempStorage);

  return 4.f * hostOut / (numSamples * numItems);
}

void Naive::calculate() {
  int* simSums = nullptr;
  cudaCheckError(
      cudaMalloc(&simSums, blocksPerGrid * threadsPerBlock * sizeof(int)));

  const int repsPerThread = numSamples;
  const dim3 gridSize(blocksPerGrid, 1, 1);
  const dim3 blockSize(threadsPerBlock, 1, 1);

  estimate_pi<<<gridSize, blockSize>>>(randStates, simSums, repsPerThread);

  float pi = cubReduce(simSums);
  cudaFree(simSums);
}

Naive::~Naive() { cudaFree(randStates); }
}  // namespace montecarlo