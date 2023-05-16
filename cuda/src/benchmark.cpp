#include "benchmark.h"

namespace benchmark {
float time_algorithm(TimedAlgorithm* algo, int numReps) {
  algo->calculate();  // warmup

  float avgTimeUs = 0;
  for (int i = 0; i < numReps; i++) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    algo->calculate();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    avgTimeUs += milliseconds * 1000 / numReps;
  }

  return avgTimeUs;
}
}  // namespace benchmark