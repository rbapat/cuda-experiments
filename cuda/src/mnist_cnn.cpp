#include "mnist_cnn.h"

namespace mnist {

CNN::CNN(int batchSize) : batchSize(batchSize) {
  writeDatasetToDevice();
  cnn = new ml::SimpleCNN(batchSize);
}

void CNN::writeDatasetToDevice() {
  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
          MNIST_DATA_LOCATION);

  numTrain = dataset.training_images.size();
  numTest = dataset.test_images.size();

  if (numTrain % batchSize != 0 || numTest % batchSize != 0) {
    throw std::runtime_error(
        "numTrain and numTest must be divisible by batch size");
  }

  populateDeviceData(numTrain, dataset.training_images, dataset.training_labels,
                     &devTrainImages, &devTrainLabels);

  populateDeviceData(numTest, dataset.test_images, dataset.test_labels,
                     &devTestImages, &devTestLabels);
}

void CNN::populateDeviceData(int numSamples,
                             std::vector<std::vector<uint8_t>> hostImages,
                             std::vector<uint8_t> hostLabels,
                             uint8_t** deviceImages, uint8_t** deviceLabels) {
  int stride = 28 * 28 * sizeof(uint8_t);

  std::vector<int> indices = std::vector<int>(numSamples);
  std::iota(indices.begin(), indices.end(), 0);
  std::random_shuffle(indices.begin(), indices.end());

  cudaCheckError(cudaMalloc((void**)deviceImages, numSamples * stride));
  cudaCheckError(
      cudaMalloc((void**)deviceLabels, numSamples * sizeof(uint8_t)));

  for (size_t i = 0; i < numSamples; i++) {
    size_t index = indices[i];
    cudaCheckError(cudaMemcpy(*deviceImages + stride * index,
                              hostImages[index].data(), stride,
                              cudaMemcpyHostToDevice));

    cudaCheckError(cudaMemcpy(*deviceLabels + sizeof(uint8_t) * index,
                              &hostLabels[index], sizeof(uint8_t),
                              cudaMemcpyHostToDevice));
  }
}

void CNN::calculate() {
  // run single epoch
  float loss = 0;
  for (int i = 0; i < numTrain; i += batchSize) {
    loss += cnn->run(devTrainImages + i, devTrainLabels + i, 0.0001);
  }

  loss /= numTrain;
  std::cout << "Loss: " << loss << std::endl;
}

std::string_view CNN::getName() { return "MNIST CNN"; }

CNN::~CNN() {
  delete cnn;
  cudaCheckError(cudaFree(devTrainImages));
  cudaCheckError(cudaFree(devTrainLabels));
  cudaCheckError(cudaFree(devTestImages));
  cudaCheckError(cudaFree(devTestLabels));
}

}  // namespace mnist