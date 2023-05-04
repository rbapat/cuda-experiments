#include "mnist_dnn.h"

namespace mnist {

cudnn_frontend::ExecutionPlan getPlanFromHeuristics(
    cudnn_frontend::OperationGraph& opGraph, cudnnHandle_t handle) {
  auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                        .setOperationGraph(opGraph)
                        .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                        .build();

  auto& engine_config =
      heuristics.getEngineConfig(heuristics.getEngineConfigCount());

  auto plan_builder = [&]() -> cudnn_frontend::ExecutionPlan {
    for (auto& ecfg : engine_config) {
      try {
        auto plan = cudnn_frontend::ExecutionPlanBuilder()
                        .setHandle(handle)
                        .setEngineConfig(ecfg, opGraph.getTag())
                        .build();
        return plan;
      } catch (cudnn_frontend::cudnnException& e) {
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
  // lets set up a 2d convolution: Conv -> bias -> relu

  // input tensor NCHW: batch_size = 1, channels = 1, height = 28, width = 28
  // must be of shape [N, num_conv_groups * input_feat_map, x, y]
  constexpr int64_t xdim[4] = {1, 1, 28, 28};
  constexpr int64_t xstrides[4] = {784, 784, 28, 1};
  auto inTns = cudnn_frontend::TensorBuilder()
                   .setDim(4, xdim)
                   .setDataType(CUDNN_DATA_FLOAT)
                   .setAlignment(4)
                   .setId('i')
                   .setStride(4, xstrides)
                   .build();
  std::cout << inTns.describe() << std::endl;

  // weight tensor NCHW: batch_size = 1, channels  = 32, kx = 3, ky = 3
  // must be of shape [out_feat_map * num_conv_groups, num_in_feat_maps, kx, ky]
  constexpr int64_t wdim[4] = {32, 1, 3, 3};
  constexpr int64_t wstrides[4] = {9, 9, 3, 1};
  auto weightTns = cudnn_frontend::TensorBuilder()
                       .setDim(4, wdim)
                       .setDataType(CUDNN_DATA_FLOAT)
                       .setAlignment(4)
                       .setId('W')
                       .setStride(4, wstrides)
                       .build();
  std::cout << weightTns.describe() << std::endl;

  // output tensor NCHW: batch_size = 1, channels = 32, height = 28, width = 28
  // must be of shape [N, num_conv_groups * out_feat_maps, height, x, y]
  constexpr int64_t ydim[4] = {1, 32, 28, 28};
  constexpr int64_t ystrides[4] = {25088, 784, 28, 1};
  auto outTns = cudnn_frontend::TensorBuilder()
                    .setDim(4, ydim)
                    .setDataType(CUDNN_DATA_FLOAT)
                    .setAlignment(4)
                    .setId('o')
                    .setStride(4, ystrides)
                    .build();
  std::cout << outTns.describe() << std::endl;

  constexpr int64_t convStride[2] = {1, 1};
  constexpr int64_t convPadding[2] = {1, 1};
  constexpr int64_t convDilation[2] = {1, 1};
  auto convDesc = cudnn_frontend::ConvDescBuilder()
                      .setComputeType(CUDNN_DATA_FLOAT)
                      .setMathMode(CUDNN_CONVOLUTION)
                      .setSpatialDimCount(2)
                      .setSpatialStride(2, convStride)
                      .setPrePadding(2, convPadding)
                      .setPostPadding(2, convPadding)
                      .setDilation(2, convDilation)
                      .build();
  std::cout << convDesc.describe() << std::endl;

  auto convOp = cudnn_frontend::OperationBuilder(
                    CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                    .setxDesc(inTns)
                    .setyDesc(outTns)
                    .setwDesc(weightTns)
                    .setcDesc(convDesc)
                    .setAlpha(1.0f)
                    .setBeta(0.0f)
                    .build();

  std::cout << convOp.describe() << std::endl;

  std::array<cudnn_frontend::Operation const*, 1> ops = {&convOp};
  auto opGraph = cudnn_frontend::OperationGraphBuilder()
                     .setHandle(handle)
                     .setOperationGraph(ops.size(), ops.data())
                     .build();

  std::cout << opGraph.describe() << " has " << opGraph.getEngineCount()
            << " possible engines" << std::endl;

  auto plan = getPlanFromHeuristics(opGraph, handle);
  execPlan = std::make_shared<cudnn_frontend::ExecutionPlan>(std::move(plan));

  cudaMalloc(&devInTns, 28 * 28 * sizeof(float));
  cudaMemset(devInTns, 1, 28 * 28 * sizeof(float));
  cudaCheckError();

  cudaMalloc(&devOutTns, 32 * 28 * 28 * sizeof(float));
  cudaMemset(devOutTns, 0, 32 * 28 * 28 * sizeof(float));
  cudaCheckError();

  cudaMalloc(&devWeightTns, 32 * 3 * 3 * sizeof(float));
  cudaMemset(devWeightTns, 0.5, 32 * 3 * 3 * sizeof(float));
  cudaCheckError();
}

void DNN::calculate() {
  void* dataPtrs[] = {devInTns, devOutTns, devWeightTns};
  int64_t uids[] = {'i', 'o', 'W'};

  auto variantPack = cudnn_frontend::VariantPackBuilder()
                         .setDataPointers(3, dataPtrs)
                         .setUids(4, uids)
                         .build();

  cudnnCheckError(cudnnBackendExecute(handle, execPlan->get_raw_desc(),
                                      variantPack.get_raw_desc()));
}

std::string_view DNN::getName() { return "cuDNN test"; }

DNN::~DNN() {}
}  // namespace mnist