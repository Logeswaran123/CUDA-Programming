#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

__global__ void PrintInfo(){
    /** 
     * Print thread idx, block idx, grid dim.
    **/
    printf("\nThreadIdx: (%d, %d, %d), BlockIdx: (%d, %d, %d), GridDim: (%d, %d, %d)",
            threadIdx.x, threadIdx.y, threadIdx.z,
            blockIdx.x, blockIdx.y, blockIdx.z,
            gridDim.x, gridDim.y, gridDim.z);
}

int main() {
    int nx, ny, nz;
    nx = 4; // number of threads along X-dim
    ny = 4; // number of threads along Y-dim
    nz = 4; // number of threads along Z-dim

    dim3 block(2, 2, 2); // 2*2*2 threads in a block
    dim3 grid(nx / block.x, ny / block.y, nz / block.z);

    PrintInfo<<<grid, block>>>();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}