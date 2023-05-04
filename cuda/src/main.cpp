#include <iostream>
#include <unordered_map>

#include "argparse.h"
#include "benchmark.h"
#include "matmul.cuh"
#include "mnist_dnn.h"
#include "montecarlo.cuh"

benchmark::TimedAlgorithm* getAlgorithm(args::Parser* ap, int argc,
                                        char* argv[]) {
  std::string_view opType = ap->parseArguments(argc, argv);

  // TODO: not important, but use enums or hashing to avoid string comparison
  if (!opType.compare("naive_matmul")) {
    return new matmul::Naive(ap->get<int>(0), ap->get<int>(1));
  } else if (!opType.compare("cublas_matmul")) {
    return new matmul::Cublas(ap->get<int>(0), ap->get<int>(1));
  } else if (!opType.compare("monte_carlo_pi")) {
    return new montecarlo::Naive(ap->get<int>(0), ap->get<int>(1),
                                 ap->get<int>(2));
  } else if (!opType.compare("mnist_dnn")) {
    return new mnist::DNN();
  }

  return nullptr;
}

int main(int argc, char* argv[]) {
  args::Parser ap;

  ap.registerOption("naive_matmul", "int(size of matrix)",
                    "int(threads per block)");
  ap.registerOption("cublas_matmul", "int(size of matrix)",
                    "int(threads per block)");
  ap.registerOption("monte_carlo_pi", "int(number of samples)",
                    "int(blocks per grid)", "int(threads per block)");
  ap.registerOption("mnist_dnn");

  auto algo = getAlgorithm(&ap, argc, argv);
  if (algo == nullptr) {
    return 0;
  }

  constexpr int numReps = 100;
  float avgTimeUs = benchmark::time_algorithm(algo, numReps);

  std::cout << algo->getName() << " took " << avgTimeUs
            << " microseconds on average" << std::endl;

  return 0;
}