# CUDA Programming 🦕🌊
[![CUDA Doc](https://img.shields.io/badge/Documentation-CUDA-green.svg)](https://docs.nvidia.com/cuda/) [![Repo parts](https://img.shields.io/badge/Repository%20Parts-Learnings-blue.svg)](https://github.com/Logeswaran123/CUDA-Programming#repository-parts)
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

#### <ins>General Explanation</ins>

* Essentially, Host runs sequential operations and Device runs parallel operations.
* The CUDA architecture is built around a scalable array of multithreaded Streaming Multiprocessors (SMs). When a CUDA program on the host CPU invokes a kernel grid, the blocks of the grid are enumerated and distributed to multiprocessors with available execution capacity. The threads of a thread block execute concurrently on one multiprocessor, and multiple thread blocks can execute concurrently on one multiprocessor. As thread blocks terminate, new blocks are launched on the vacated multiprocessors. Refer [thebeardsage/cuda-streaming-multiprocessors](http://thebeardsage.com/cuda-streaming-multiprocessors/) for detailed explanation.
* Multiple thread blocks can execute on same single Streaming Multiprocessor, but one thread block cannot execute on multiple Streaming Multiprocessors.
* The maximum x, y and z dimensions of a block are 1024, 1024 and 64, and it should be allocated such that x × y × z ≤ 1024, which is the maximum number of threads per block. Blocks can be organized into one, two or three-dimensional grids of up to 2^31-1, 65,535 and 65,535 blocks in the x, y and z dimensions respectively. Unlike the maximum threads per block, there is not a blocks per grid limit distinct from the maximum grid dimensions.

#### <ins>Warps</ins>

* A warp is the basic unit of execution in a cuda program. A warp is a set of 32 threads within a thread block such that all the threads in a warp execute the same instruction. These threads are selected serially by the Streaming Multiprocessor.
* If a set of threads execute different instruction compared to other threads of a warp, then warp divergence occurs. This can reduce performance of the cuda program.

#### <ins>Dynamic Parallelism</ins>
 [![Dynamic Parallelism](https://img.shields.io/badge/Dynamic%20Parallelism-Blog-green.svg)](https://developer.nvidia.com/blog/introduction-cuda-dynamic-parallelism/)
<br />

* Early CUDA programs have been designed in a way that GPU workload was completely in control of Host thread. Programs had to perform a sequence of kernel launches, and for best performance each kernel had to expose enough parallelism to efficiently use the GPU.
* CUDA 5.0 introduced Dynamic Parallelism, which makes it possible to launch kernels from threads running on the device; threads can launch more threads. An application can launch a coarse-grained kernel which in turn launches finer-grained kernels to do work where needed. This avoids unwanted computations while capturing all interesting details.
* This reduces the need to transfer control and data between host and GPU device.
* Kernel executions are classified into Parent and Child grids. Parent grid start execution and dispatches some workload to child grid. Parent grid end the execution when kernel execution is complete. A child grid inherits from the parent grid certain attributes and limits, such as the L1 cache / shared memory configuration and stack size.
* Grid launches in a device thread is visible across all threads in the thread block. Execution of a thread block is not complete untill all child threads created in the block are complete.
* Grids launched with dynamic parallelism are fully nested. This means that child grids always complete before the parent grids that launch them, even if there is no explicit synchronization

## Images
#### Schematic
![alt text](https://github.com/Logeswaran123/CUDA-Programming/blob/main/images/schematic.jpg "Schematic")

#### Software vs. Hardware perspective
![alt text](https://github.com/Logeswaran123/CUDA-Programming/blob/main/images/SoftwarevsHardware.jpg "Software vs. Hardware perspective")

## Tools
* [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) <br />
NVIDIA Nsight™ Systems is a system-wide performance analysis tool designed to visualize an application’s algorithms, help you identify the largest opportunities to optimize, and tune to scale efficiently across any quantity or size of CPUs and GPUs, from large servers to our smallest system on a chip (SoC).
* [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) <br />
NVIDIA® Nsight™ Compute is an interactive kernel profiler for CUDA applications. It provides detailed performance metrics and API debugging via a user interface and command line tool. In addition, its baseline feature allows users to compare results within the tool. Nsight Compute provides a customizable and data-driven user interface and metric collection and can be extended with analysis scripts for post-processing results.

## Repository Parts
| Number |   Repository   |  Description |
|:-----------------:|----------------|--------------|
| 1 | [Hello World](https://github.com/Logeswaran123/CUDA-Programming/tree/main/1_hello_world) | Programmer's induction. Hello World from GPU |
| 2 | [Print](https://github.com/Logeswaran123/CUDA-Programming/tree/main/2_print) | Print ThreadIdx, BlockIdx, GridDim |
| 3 | [Addition](https://github.com/Logeswaran123/CUDA-Programming/tree/main/3_add) | Perform addition operation on GPU |
| 4 | [Add Arrays](https://github.com/Logeswaran123/CUDA-Programming/tree/main/4_add_arrays) | Perform addition of three arrays on GPU |
| 5 | [Global Index](https://github.com/Logeswaran123/CUDA-Programming/tree/main/5_gid_calculation) | Calculate Global Index for any dimensional grid and any dimensional block |
| 6 | [Device properties](https://github.com/Logeswaran123/CUDA-Programming/tree/main/6_device_properties) | Print some GPU device properties |
| 7a | [Reduce Sum with Loop Unroll](https://github.com/Logeswaran123/CUDA-Programming/tree/main/7_reduction_loop_warp_complete/reduction_loop_unroll) | Perform reduction sum operation with loop unroll in GPU kernel |
| 7b | [Reduce Sum with Warp Unroll]() | TODO |
| 7c | [Reduce Sum with Complete Unroll]() | TODO |

## Terms
|  |  |  |  |  |  |  |  |  |  |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Streaming Multiprocessor | Grid | Thread Block | Thread | Warp | Kernel | _syncthread | Occupancy | Shared memory | Registers |
| Dynamic parallelism | Parallel reduction | Parent | Child |  |  |  |  |  |  |

## References
* [CUDA Programming Masterclass with C++](https://www.udemy.com/course/cuda-programming-masterclass/)
* [CUDA Thread Indexing cheat sheet](https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf)
* [CUDA Refresher Blog](https://developer.nvidia.com/blog/tag/cuda-refresher/)

Happy Learning! 😄
