#include <iostream>
#include <unordered_map>

#include "argparse.h"
#include "matmul.cuh"

int main(int argc, char* argv[]) {
    args::Parser ap;

    ap.registerOption("matmul", "int(number of times to execute)", "int(size of matrix)", "int(threads per block)");
    auto opType = ap.parseArguments(argc, argv);
    
    // TODO: use enums or hashing to avoid string comparison
    if (!opType.compare("matmul")) {
        float avgTimeUs = matmul::naive::time_execution(ap.get<int>(0), ap.get<int>(1), ap.get<int>(2));
        std::cout << "Operation took " << avgTimeUs << " us on average" << std::endl;
    }

    return 0;
}