#include <iostream>
#include <unordered_map>

#include "argparse.h"

void matmul(int numThreads, int numBlocks) {
    std::cout << "Doing matmul with " << numThreads << " threads and " << numBlocks << " blocks" << std::endl;
}

int main(int argc, char* argv[]) {
    args::Parser ap;

    ap.registerOption("matmul", "int(number of threads)", "int(number of blocks)");
    auto opType = ap.parseArguments(argc, argv);
    
    // TODO: use enums or hashing to avoid string comparison
    if (!strcmp(opType, "matmul")) {
        matmul(ap.get<int>(0), ap.get<int>(1));
    }

    return 0;
}