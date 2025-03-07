# Flash Attention CUDA Implementation

## Overview
This project implements Flash Attention forward and backward pass using CUDA. Flash Attention is an efficient attention mechanism designed to optimize memory access patterns and improve performance over traditional attention mechanisms.

## Features
- **CUDA kernels for Flash Attention**: Implements memory-efficient forward and backward passes.
- **Optimized parallelization**: Uses shared memory, thread cooperation, and warp-level optimizations.
- **Efficient computation**: Reduces redundant computations with a block-based processing approach.

## Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C++ compiler supporting CUDA

## Compilation & Execution
To compile and run the program, use the following commands:

```sh
nvcc flash_attention.cu -o flash_attention -arch=sm_70 -std=c++11
./flash_attention
```
Replace `sm_70` with the appropriate compute capability for your GPU.

## File Structure
- `flash_attention.cu` : Main CUDA implementation file containing the forward and backward kernels.
- `README.md` : This documentation file.

## Implementation Details
### Forward Pass
The forward pass computes the scaled dot-product attention in a block-wise manner:
1. Loads query (`Q`), key (`K`), and value (`V`) matrices into shared memory.
2. Computes scaled dot-product attention scores (`S = QK^T / sqrt(d)`).
3. Applies softmax using log-sum-exp trick.
4. Computes weighted sum to obtain the output (`O = softmax(S) * V`).
5. Stores the log-sum-exp values for backward computation.

### Backward Pass
The backward pass computes gradients with respect to `Q`, `K`, and `V`:
1. Computes `D`, the row-wise sum of element-wise product of `dO` and `O`.
2. Computes `dP = dO * V^T`.
3. Computes `dS` using `dP ⊙ P - P ⊙ (P^T · dP)`.
4. Computes `dQ = dS * K`, `dK = dS^T * Q`, and `dV = P^T * dO`.

## Example Usage
The `main` function initializes random input tensors, runs the forward pass, and computes gradients via the backward pass. It prints the first 10 values of the computed gradients for verification.

## Performance Optimizations
- **Shared Memory Usage**: Reduces global memory accesses by storing intermediate computations in shared memory.
- **Parallel Softmax Computation**: Uses log-sum-exp trick to stabilize computations.
- **Atomic Operations**: Ensures correct gradient accumulation using `atomicAdd`.
- **Memory Coalescing**: Optimizes memory accesses to improve performance.

## Sample Output
```
Running forward pass...
Running backward pass...
Gradient dQ (first 10 values): 0.123 0.456 ...
Gradient dK (first 10 values): 0.789 0.012 ...
Gradient dV (first 10 values): 0.345 0.678 ...
```

## References
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

## License
This project is open-source and available under the MIT License.
