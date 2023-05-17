# cuda-experiments

## Final Project for COMPSCI532

- `python` folder contains several CPU and GPU experiments
  - Logistic Regression, CuPy
  - Logistic Regression, PyTorch
  - Matrix Multiplication, CuPy
  - Matrix Multiplication, PyTorch
  - Monte Carlo Estimation, CuPy
  - Monte Carlo Estimation, PyTorch
  - MNIST Conv Net, PyTorch
- `notes` contains some preliminary notes I used to learn CUDA
- `data` folder contains MNIST data
- `cuda` folder contains my experiments with cuda. When compiling it, make sure CUB and cuDNN are installed, along with the rest of the CUDA toolkit. run `./bin/out` to get more informationa about what different operations are available. I implemented:
  - Argument parser and benchmarking code
  - CUDA optimized monte carlo estimation
  - Naive CUDA matrix multiplication
  - cuBLAS matrix multiplication
  - CUDA optimized logistic regression on Iris flower dataset
  - convolutional neural network using cuDNN.
    - currently, it keeps computing NaNs, but it can be timed/benchmarked. I'm not sure if the benchmark is accurate though.
