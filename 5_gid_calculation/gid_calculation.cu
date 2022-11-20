#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.cuh"

#include <iostream>
#include <numeric>

__global__ void GIDCalculation(int* input, int arraysize) {
    /** 
     * Calculate Global Index for any dimensional grid and any dimensional block.
     * Reference: https://forums.developer.nvidia.com/t/calculate-global-thread-id/23541/4
    **/

    int columnsize = threadIdx.x;
    int rowsize = threadIdx.y;
    int aislesize = threadIdx.z;
    int threads_per_row = blockDim.x;   // Threads in X-direction
    int threads_per_aisle = blockDim.x * blockDim.y;  // Threads in X-direction & Y-direction for total threads per aisle
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;

    int rowOffset = rowsize * threads_per_row;    // How many rows to offset
    int aisleOffset = aislesize * threads_per_aisle;  // How many aisles to offset

    int blockColumn = blockIdx.x;
    int blockRow = blockIdx.y;
    int blockAisle = blockIdx.z;
    int blocks_per_row = gridDim.x; // Blocks in X-direction
    int blocks_per_aisle = gridDim.x * gridDim.y; // Blocks in X-direction & Y-direction for total blocks per aisle
    int blockRowOffset = blockRow * blocks_per_row;   // How many rows to Block offset
    int blockAisleOffset = blockAisle * blocks_per_aisle; // How many aisles to Block offset
    int blockId = blockColumn + blockRowOffset + blockAisleOffset;

    int blockOffset = (blockId * threads_per_block);

    int tid = rowOffset + aisleOffset + columnsize;
    int gid = blockOffset + aisleOffset + rowOffset + columnsize;

    if (gid < arraysize) {
        printf("\nblockIdx : (%d,%d,%d), ThreadIdx : (%d,%d,%d), tid : (%2d), gid : (%2d), input[gid] : (%2d)", 
            blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, tid, gid, input[gid]);
    }
}

int main() {
    const size_t arraysize = 100;
    int host_arr[arraysize];
    std::iota(host_arr, host_arr + arraysize, 1);

    // Print data on host
    printf("\n-----Print from Host-----\n");
    for (size_t i = 0; i < arraysize; i++) {
        if ((i >= 32) && (i%32 == 0)) {
            printf("\n%3d ", host_arr[i]);
        }
        else {
            printf("%3d ", host_arr[i]);
        }
    }

    int *device_arr;
    GPUErrorCheck(cudaMalloc((void**)&device_arr, sizeof(int) * arraysize));
    GPUErrorCheck(cudaMemcpy(device_arr, host_arr, sizeof(int) * arraysize, cudaMemcpyHostToDevice));

    // Set block and grid size
    int nx = 2;
    int ny = 5;
    int nz = 10;
    dim3 block(2, 2, 2);
    dim3 grid((nx / block.x) + 1, (ny / block.y) + 1, (nz / block.z) + 1);

    // Launch kernel
    // Print data on device
    printf("\n\n-----Print from Device-----");
    GIDCalculation<<<grid, block>>>(device_arr, arraysize);
    GPUErrorCheck(cudaDeviceSynchronize());

    GPUErrorCheck(cudaFree(device_arr));
    GPUErrorCheck(cudaDeviceReset());
    return 0;
}