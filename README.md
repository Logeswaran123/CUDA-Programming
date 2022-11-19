# CUDA Programming 🦕🌊
My CUDA Programming learning journey.


## Basics
The basic flow in CUDA programming is,
1. Initialize data from CPU.
2. Transfer data from CPU to GPU.
3. Lauch necessary kernel executions on data.
4. Tranfer data from GPU to CPU.
5. Release memory from CPU and GPU.

![alt text](https://github.com/Logeswaran123/CUDA-Programming/blob/main/images/schematic.jpg "Schematic")

### Terms usage
|   Term   |  Meaning |
|----------|----------|
| SISD | Single Instruction Single Data |
| SIMD | Single Instruction Multiple Data |
| MISD | Multiple Instruction Single Data |
| MIMD | Multiple Instruction Multiple Data |
| SIMT | Single Instruction Multiple Threads |

<br />

|   Term   |  Meaning |
|----------|----------|
| Host | CPU |
| Device | GPU |

<br />

Essentially, Host runs sequential operations and Device runs parallel operations.

General syntax for launching a kernel, <br />
```
kernel_name<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(arguments);
```
```
kernel_name<<<GRID, BLOCK>>>(arguments);
```
<br />
The maximum x, y and z dimensions of a block are 1024, 1024 and 64, and it should be allocated such that x × y × z ≤ 1024, which is the maximum number of threads per block. Blocks can be organized into one, two or three-dimensional grids of up to 2^31-1, 65,535 and 65,535 blocks in the x, y and z dimensions respectively. Unlike the maximum threads per block, there is not a blocks per grid limit distinct from the maximum grid dimensions.
