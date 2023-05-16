#include "mnist_dnn.h"

namespace mnist {

int64_t makeId(char parentName, char opName, char varName) {
  return (parentName << 16) + (opName << 8) + varName;
}

cudnn_frontend::ExecutionPlan getPlanFromHeuristics(
    cudnn_frontend::OperationGraph& opGraph, cudnnHandle_t handle) {
  auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                        .setOperationGraph(opGraph)
                        .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                        .build();

  std::cout << "got heur: " << heuristics.getEngineConfigCount() << std::endl;
  auto& engine_config =
      heuristics.getEngineConfig(heuristics.getEngineConfigCount());

  std::cout << "engine config" << std::endl;
  auto plan_builder = [&]() -> cudnn_frontend::ExecutionPlan {
    for (auto& ecfg : engine_config) {
      try {
        auto plan = cudnn_frontend::ExecutionPlanBuilder()
                        .setHandle(handle)
                        .setEngineConfig(ecfg, opGraph.getTag())
                        .build();
        return plan;
      } catch (cudnn_frontend::cudnnException& e) {
        std::cout << e.what() << std::endl;
        continue;
      }
    }
    return cudnn_frontend::ExecutionPlanBuilder()
        .setHandle(handle)
        .setEngineConfig(engine_config[0], opGraph.getTag())
        .build();
  };

  return plan_builder();
}

DNN::DNN() {
  cudnnCheckError(cudnnCreate(&handle));

  // TODO: I really hate how i'm passing the id's, this is just a super messy
  // and hacky solution. Will refactor and clean eventually.

  auto convOp1 = ml::fwd::Conv2d(
      1, 1, 28, 28, 3, 32, '1', false, true, makeId('1', CONV_OP_ID, 'i'),
      makeId('1', CONV_OP_ID, 'o'));  // 1[Ci, Co, Cw]

  auto biasOp1 = ml::fwd::Bias(1, 32, 28, 28, '1', true, true,
                               makeId('1', CONV_OP_ID, 'o'),
                               makeId('1', BIAS_OP_ID, 'o'));  // 1[Bi, Bo, Bb]

  auto reluOp1 = ml::fwd::ReLU(1, 32, 28, 28, '1', true, true,
                               makeId('1', BIAS_OP_ID, 'o'),
                               makeId('1', RELU_OP_ID, 'o'));  // 1[Ri, Ro]

  auto convOp2 = ml::fwd::Conv2d(
      1, 32, 28, 28, 3, 64, '2', true, true, makeId('1', RELU_OP_ID, 'o'),
      makeId('2', CONV_OP_ID, 'o'));  // 2[Ci, Co, Cw]

  auto biasOp2 = ml::fwd::Bias(1, 64, 28, 28, '2', true, true,
                               makeId('2', CONV_OP_ID, 'o'),
                               makeId('2', BIAS_OP_ID, 'o'));  // 2[Bi, Bo, Bb]

  auto reluOp2 = ml::fwd::ReLU(1, 64, 28, 28, '2', true, true,
                               makeId('2', BIAS_OP_ID, 'o'),
                               makeId('2', RELU_OP_ID, 'o'));  // 2[Ri, Ro]

  auto poolOp1 = ml::fwd::Pool(1, 64, 28, 28, 2, 2, CUDNN_RESAMPLE_MAXPOOL, '3',
                               true, false, makeId('2', RELU_OP_ID, 'o'),
                               makeId('3', POOL_OP_ID, 'o'));  //  3[Pi, Po]

  // std::array<cudnn_frontend::Operation const*, 7> ops = {
  //     &convOp1, &biasOp1, &reluOp1, &convOp2, &biasOp2, &reluOp2, &poolOp1};

  std::array<cudnn_frontend::Operation const*, 2> ops = {
      &convOp1, &biasOp1};  //, &biasOp1, &reluOp1, &convOp2, &biasOp2,
                            //&reluOp2, &poolOp1};

  auto opGraph = cudnn_frontend::OperationGraphBuilder()
                     .setHandle(handle)
                     .setOperationGraph(ops.size(), ops.data())
                     .build();

  std::cout << opGraph.describe() << std::endl;

  std::cout << "Getting plan...\n" << std::endl;
  auto plan = getPlanFromHeuristics(opGraph, handle);
  plan.describe();
  execPlan = std::make_shared<cudnn_frontend::ExecutionPlan>(std::move(plan));

  std::cout << "Allocating..." << std::endl;
  cudaMalloc(&devInput, 28 * 28 * sizeof(float));
  cudaMemset(devInput, 1, 28 * 28 * sizeof(float));
  cudaCheckError();

  std::cout << "1..." << std::endl;
  cudaMalloc(&devConvWeight1, 32 * 3 * 3 * sizeof(float));
  cudaMemset(devConvWeight1, 0.5, 32 * 3 * 3 * sizeof(float));
  cudaCheckError();

  std::cout << "2..." << std::endl;
  cudaMalloc(&devBias1, 32 * sizeof(float));
  cudaMemset(devBias1, 0.5, 32 * sizeof(float));
  cudaCheckError();

  std::cout << "3..." << std::endl;
  cudaMalloc(&devConvWeight2, 64 * 32 * 3 * 3 * sizeof(float));
  cudaMemset(devConvWeight2, 0.5, 64 * 32 * 3 * 3 * sizeof(float));
  cudaCheckError();

  std::cout << "4..." << std::endl;
  cudaMalloc(&devBias2, 64 * sizeof(float));
  cudaMemset(devBias2, 0.5, 64 * sizeof(float));
  cudaCheckError();

  std::cout << "5..." << std::endl;
  cudaMalloc(&devPoolOut, 64 * 14 * 14 * sizeof(float));
  cudaMemset(devPoolOut, 0.5, 64 * 14 * 14 * sizeof(float));
  cudaCheckError();
  std::cout << "Done" << std::endl;
}

void DNN::calculate() {
  std::cout << "Creating packs..." << std::endl;
  void* dataPtrs[] = {devInput,       devConvWeight1, devBias1,
                      devConvWeight2, devBias2,       devPoolOut};
  int64_t uids[] = {'1Ci', '1Cw', '1Bb', '2Cw', '2Bb', '3Po'};

  std::cout << "Packing..." << std::endl;
  auto variantPack = cudnn_frontend::VariantPackBuilder()
                         .setDataPointers(6, dataPtrs)
                         .setUids(6, uids)
                         .build();

  std::cout << "Executing..." << std::endl;
  cudnnCheckError(cudnnBackendExecute(handle, execPlan->get_raw_desc(),
                                      variantPack.get_raw_desc()));
}

std::string_view DNN::getName() { return "cuDNN test"; }

DNN::~DNN() {}
}  // namespace mnist