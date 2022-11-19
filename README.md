# CUDA Programming 🦕🌊
[![CUDA Doc](https://img.shields.io/badge/Documentation-CUDA-green.svg)](https://docs.nvidia.com/cuda/)
<br /><br />
My CUDA Programming learning journey.


## Basics
The basic flow in CUDA programming is,
1. Initialize data from CPU.
2. Transfer data from CPU to GPU.
3. Lauch necessary kernel executions on data.
4. Tranfer data from GPU to CPU.
5. Release memory from CPU and GPU.

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
| SM | Streaming Multiprocessor |

<br />

* Essentially, Host runs sequential operations and Device runs parallel operations.
* The CUDA architecture is built around a scalable array of multithreaded Streaming Multiprocessors (SMs). When a CUDA program on the host CPU invokes a kernel grid, the blocks of the grid are enumerated and distributed to multiprocessors with available execution capacity. The threads of a thread block execute concurrently on one multiprocessor, and multiple thread blocks can execute concurrently on one multiprocessor. As thread blocks terminate, new blocks are launched on the vacated multiprocessors. Refer [thebeardsage/cuda-streaming-multiprocessors](http://thebeardsage.com/cuda-streaming-multiprocessors/) for detailed explanation.
* Multiple thread blocks can execute on same single Streaming Multiprocessor(SM), but one thread block cannot execute on multiple Streaming Multiprocessors.
* The maximum x, y and z dimensions of a block are 1024, 1024 and 64, and it should be allocated such that x × y × z ≤ 1024, which is the maximum number of threads per block. Blocks can be organized into one, two or three-dimensional grids of up to 2^31-1, 65,535 and 65,535 blocks in the x, y and z dimensions respectively. Unlike the maximum threads per block, there is not a blocks per grid limit distinct from the maximum grid dimensions.

<br />

![alt text](https://github.com/Logeswaran123/CUDA-Programming/blob/main/images/schematic.jpg "Schematic")

### General syntax 
For launching a kernel, <br />
```
kernel_name<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(arguments);
```
```
kernel_name<<<GRID, BLOCK>>>(arguments);
```
<br />
