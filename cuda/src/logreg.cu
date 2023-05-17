#include <cub/cub.cuh>

#include "logreg.cuh"

namespace iris {
LogisticRegression::LogisticRegression(int numEpoch) : numEpoch(numEpoch) {
  numSamples = loadData(&devDataPtr, &devLabelPtr);
  if (numSamples == 0) {
    throw std::runtime_error("Unable to parse Iris dataset\n");
  }

  cublasCheckError(cublasCreate(&handle));
}

int LogisticRegression::loadData(void** devDataPtr, void** devLabelPtr) {
  std::ifstream irisFile(IRIS_DATA_FILE);
  if (!irisFile.is_open()) {
    throw std::runtime_error("unable to open iris file");
  }

  int numLines = 0;
  std::string line;
  while (std::getline(irisFile, line)) ++numLines;
  irisFile.clear();
  irisFile.seekg(0);

  constexpr size_t stride = sizeof(float) * 4;
  float* hostDataPtr = (float*)calloc(numLines, stride);
  uint8_t* hostLabelPtr = (uint8_t*)calloc(numLines, sizeof(uint8_t));

  int index = 0;
  while (std::getline(irisFile, line)) {
    std::stringstream ss(line);
    std::string attribute;

    for (size_t featureIdx = 0; featureIdx < 4; featureIdx++) {
      std::getline(ss, attribute, ',');
      hostDataPtr[index * 4 + featureIdx] = std::stof(attribute);
    }

    hostLabelPtr[index] = attribute.compare("Iris-setosa") ? 0 : 1;
    index++;
  }

  cudaCheckError(cudaMalloc(devDataPtr, numLines * stride));
  cudaCheckError(cudaMemcpy(*devDataPtr, hostDataPtr, numLines * stride,
                            cudaMemcpyHostToDevice));
  cudaCheckError(cudaMalloc(devLabelPtr, numLines * sizeof(uint8_t)));
  cudaCheckError(cudaMemcpy(*devLabelPtr, hostLabelPtr,
                            numLines * sizeof(uint8_t),
                            cudaMemcpyHostToDevice));

  free(hostDataPtr);
  free(hostLabelPtr);
  return numLines;
}

__global__ void calcGradAndLoss(int numSamples, float* sigmoidIn,
                                float* devDiff, float* devLosses,
                                uint8_t* devLabels) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  while (idx < numSamples) {
    uint8_t y = devLabels[idx];
    float y_hat = 1 / (1 + expf(sigmoidIn[idx] * -1));

    devDiff[idx] = y_hat - y;
    devLosses[idx] =
        (y * logf(y_hat) + (1 - y) * logf(1 - y_hat)) / -numSamples;
    idx += gridDim.x * blockDim.x;
  }
}

void LogisticRegression::calculate() {
  float rands[4];
  for (int i = 0; i < 4; i++) rands[i] = 1e-4 * (float)rand() / RAND_MAX;

  float* devW;
  cudaCheckError(cudaMalloc(&devW, 4 * sizeof(float)));
  cudaCheckError(
      cudaMemcpy(devW, rands, 4 * sizeof(float), cudaMemcpyHostToDevice));

  float* devGrads;
  cudaCheckError(cudaMalloc(&devGrads, 4 * sizeof(float)));

  float* sigmoidIn;
  cudaCheckError(cudaMalloc(&sigmoidIn, numSamples * sizeof(float)));

  float* devLosses;
  cudaCheckError(cudaMalloc(&devLosses, numSamples * sizeof(float)));

  float* devDiff;
  cudaCheckError(cudaMalloc(&devDiff, numSamples * sizeof(float)));

  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  const float nu = -LEARNING_RATE / numSamples;

  // set up cub
  // float* deviceOut;
  // cudaCheckError(cudaMalloc(&deviceOut, sizeof(float)));

  // void* tempStorage = NULL;
  // size_t tempStorageSize = 0;

  // CubDebugExit(cub::DeviceReduce::Sum(tempStorage, tempStorageSize,
  // devLosses,
  //                                     deviceOut, numSamples));

  // cudaCheckError(cudaMalloc(&tempStorage, tempStorageSize));

  for (int epoch = 0; epoch < numEpoch; epoch++) {
    cublasCheckError(cublasSgemv_v2(
        handle, CUBLAS_OP_T, 4, numSamples, &alpha, (const float*)devDataPtr, 4,
        (const float*)devW, 1, &beta, (float*)sigmoidIn, 1));

    dim3 gridDim((numSamples / 128) + 1);
    dim3 blockDim(128);
    calcGradAndLoss<<<gridDim, blockDim>>>(numSamples, sigmoidIn, devDiff,
                                           devLosses, (uint8_t*)devLabelPtr);

    cublasCheckError(cublasSgemv_v2(
        handle, CUBLAS_OP_N, 4, numSamples, &alpha, (const float*)devDataPtr, 4,
        (const float*)devDiff, 1, &beta, (float*)devGrads, 1));

    // Saxpy is probably overkill
    cublasCheckError(cublasSaxpy_v2(handle, 4, &nu, (const float*)devGrads, 1,
                                    (float*)devW, 1));

    // calculate loss for epoch
    // CubDebugExit(cub::DeviceReduce::Sum(tempStorage, tempStorageSize,
    // devLosses,
    //                                     deviceOut, numSamples));

    // float hostOut = 0;
    // cudaCheckError(
    //     cudaMemcpy(&hostOut, deviceOut, sizeof(float),
    //     cudaMemcpyDeviceToHost));

    // std::cout << "[" << epoch << "] Loss: " << hostOut << std::endl;
  }

  cudaCheckError(cudaFree(devW));
}
std::string_view LogisticRegression::getName() {
  return "IRIS Logistic Regression";
}

LogisticRegression::~LogisticRegression() {
  cudaCheckError(cudaFree(devDataPtr));
  cudaCheckError(cudaFree(devLabelPtr));
  cublasCheckError(cublasDestroy(handle));
}

}  // namespace iris