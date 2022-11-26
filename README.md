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
* Early CUDA programs have been designed in a way that GPU workload was completely in control of Host thread. Programs had to perform a sequence of kernel launches, and for best performance each kernel had to expose enough parallelism to efficiently use the GPU.
* CUDA 5.0 introduced Dynamic Parallelism, which makes it possible to launch kernels from threads running on the device; threads can launch more threads. An application can launch a coarse-grained kernel which in turn launches finer-grained kernels to do work where needed. This avoids unwanted computations while capturing all interesting details.
* This reduces the need to transfer control and data between host and GPU device.
* Kernel executions are classified into Parent and Child grids. Parent grid start execution and dispatches some workload to child grid. Parent grid end the execution when kernel execution is complete. A child grid inherits from the parent grid certain attributes and limits, such as the L1 cache / shared memory configuration and stack size.
* Grid launches in a device thread is visible across all threads in the thread block. Execution of a thread block is not complete untill all child threads created in the block are complete.
* Grids launched with dynamic parallelism are fully nested. This means that child grids always complete before the parent grids that launch them, even if there is no explicit synchronization

---

### CUDA Memory Management
Refer [CUDA Runtime API documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY) for details.

#### <ins>Registers (On-Chip)</ins>
 [![Registers](https://img.shields.io/badge/Registers-Blog-white.svg)](https://carpentries-incubator.github.io/lesson-gpu-programming/06-global_local_memory/index.html#:~:text=CUDA%20programming%20model.-,Registers,-Registers%20are%20fast)
* Registers are fast on-chip memories that are used to store operands for the operations executed by the computing cores.
* In general all scalar variables defined in CUDA code are stored in registers. 
* Registers are local to a thread, and each thread has exclusive access to its own registers. Values in registers cannot be accessed by other threads, even from the same block, and are not available for the host. Registers are also not permanent, therefore data stored in registers is only available during the execution of a thread.
* <b>Register Spills:</b> If a kernel uses more registers than the hardware limit, the excess registers will spill over to local memory causing performance deterioration.

#### <ins>Shared Memory (On-Chip)</ins>
[![Shared Memory](https://img.shields.io/badge/Shared%20Memory-Blog-white.svg)](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
* On-chip memory shared/partitioned among thread blocks. Lifetime is lifetime of execution of the thread block.
* Shared memory is allocated per thread block, so all threads in the block have access to the same shared memory. Threads can access data in shared memory loaded from global memory by other threads within the same thread block.

#### <ins>Local Memory</ins>
 [![Local Memory](https://img.shields.io/badge/Local%20Memory-Blog-white.svg)](https://carpentries-incubator.github.io/lesson-gpu-programming/06-global_local_memory/index.html#:~:text=the%20kernel%20terminates.-,Local%20Memory,-Memory%20can%20also)
* Variables that cannot be stored in register space are stored in local memory. Memory that cannot be decided at compile time are stored in local memory.
* Memory can also be statically allocated from within a kernel, and according to the CUDA programming model such memory will not be global but local memory.
* Local memory is only visible, and therefore accessible, by the thread allocating it. So all threads executing a kernel will have their own privately allocated local memory.

There are other types of memory: Global, Constant, Texture. Refer [CUDA Memory Model](https://www.3dgep.com/cuda-memory-model/#CUDA_Memory_Types) for details.

---

#### <ins>Pinned memory</ins>
 [![Memory Paging](https://img.shields.io/badge/Memory%20Paging-Wiki-white.svg)](https://en.wikipedia.org/wiki/Memory_paging)
* CUDA uses DMA to transfer pinned memory to GPU device. Pageable memory cannot be directly transfered to device. So, first it's copied to pinned (page-locked) memory and then copied to GPU device.
* Pinned to Device transfer is faster than Pageable to Device transfer.
* <b>cudaMallocHost</b> and <b>cudaFreeHost</b> functions can be used to allocate pinned memory directly.
* Refer [Nvidia blog](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/#:~:text=execution%20time%20correspondingly.-,Pinned%20Host%20Memory,-Host%20(CPU)%20data) for details.

#### <ins>Global Memory Access Patterns</ins>
* Refer [Nvidia blog](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/).
* Refer [Medium article](https://medium.com/distributed-knowledge/cuda-memory-management-use-cases-f9d340f7c704).
* First memory acccess is L1 cache access (termed as normal cached memory access). When memory request comes to L1 cache, and L1 cache misses, then the request will be sent to L2 cache. If L2 cache misses, then the request will be sent to DRAM. Memory load that doesn't use L1 cache are referred to as un-cached memory acccess.
* If L1 cache line is used, then the memory is served in 128 bytes segment.
* If L2 cache is only used, then the memory is served in 32 bytes segment.

#### <ins>Global Memory Store</ins>
* In memory write, only L2 cache is used. It is divided into 32 bytes segment.

#### <ins>Array of Structures (AOS) vs. Structure of Arrays (SOA)</ins>
* Refer [Wiki](https://en.wikipedia.org/wiki/AoS_and_SoA#:~:text=to%20memory%20coalescing.-,Software%20support,support%20a%20data%2Doriented%20design.) for explanation on AOS and SOA.
* In CUDA programming, SOA is preferred over AOS for global memory efficiency. This because, in SOA, the array is stored in coalesced fashion reducing number of memory transactions.

#### <ins>Partition Camping</ins>
 [![Partition Camping](https://img.shields.io/badge/Partition%20Camping%20and%20Matrix%20Transpose-Slides-white.svg)](https://www.csd.uwo.ca/~mmorenom/HPC-Slides/Optimizing_CUDA_Code-2x2.pdf)
 * Partition camping occurs when global memory accesses are directed through a subset of partitions, causing requests to queue up at some partitions while other partitions go unused.
 * Since partition camping concerns how active thread blocks behave,
the issue of how thread blocks are scheduled on multiprocessors is
important.

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
| 1 | [Hello World](https://github.com/Logeswaran123/CUDA-Programming/tree/main/1_hello_world) | Programmer's induction. Hello World from GPU. |
| 2 | [Print](https://github.com/Logeswaran123/CUDA-Programming/tree/main/2_print) | Print ThreadIdx, BlockIdx, GridDim. |
| 3 | [Addition](https://github.com/Logeswaran123/CUDA-Programming/tree/main/3_add) | Perform addition operation on GPU. |
| 4 | [Add Arrays](https://github.com/Logeswaran123/CUDA-Programming/tree/main/4_add_arrays) | Perform addition of three arrays on GPU. |
| 5 | [Global Index](https://github.com/Logeswaran123/CUDA-Programming/tree/main/5_gid_calculation) | Calculate Global Index for any dimensional grid and any dimensional block. |
| 6 | [Device properties](https://github.com/Logeswaran123/CUDA-Programming/tree/main/6_device_properties) | Print some GPU device properties. |
| 7a | [Reduce Sum with Loop Unroll](https://github.com/Logeswaran123/CUDA-Programming/tree/main/7_reduction_loop_warp_complete/reduction_loop_unroll) | Perform reduction sum operation with loop unroll in GPU kernel. |
| 7b | [Reduce Sum with Warp Unroll](https://github.com/Logeswaran123/CUDA-Programming/tree/main/7_reduction_loop_warp_complete/reduction_warp_unroll) | Perform reduction sum operation with warp unroll in GPU kernel.<br />Solution for warp divergence. |
| 7c | [Reduce Sum with Complete Unroll](https://github.com/Logeswaran123/CUDA-Programming/tree/main/7_reduction_loop_warp_complete/reduction_complete_unroll) | Perform reduction sum operation with completely unrolled loop in GPU kernel. |
| 8 | [Coalesced vs. Un-Coalesced memory pattern]() | TODO |
| 9 | [Matrix Transpose](https://github.com/Logeswaran123/CUDA-Programming/tree/main/9_matrix_transpose) | Perform Matrix transpose in different fashion. |

## Terms
|  |  |  |  |  |  |  |  |  |  |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Streaming Multiprocessor | Grid | Thread Block | Thread | Warp | Kernel | _syncthread | Occupancy | Shared memory | Registers |
| Dynamic parallelism | Parallel reduction | Parent | Child | Temporal locality | Spatial locality | Coalesced memory pattern | Un-Coalesced memory pattern | L1 Cache  | L2 Cache |
|  |  |  |  |  |  |  |  |  |  |

## References
* [CUDA Programming Masterclass with C++](https://www.udemy.com/course/cuda-programming-masterclass/)
* [CUDA Thread Indexing cheat sheet](https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf)
* [CUDA Refresher Blog](https://developer.nvidia.com/blog/tag/cuda-refresher/)

Happy Learning! 😄
