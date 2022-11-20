#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        printf("\nNo device found with compute capability >=2.0 that are available for execution.");
    }

    int device_number = 0;
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_number);

    printf("\n-----Device Properties-----\n");
    printf("\nDevice name: %s", device_prop.name);
    printf("\nCompute capability: %d.%d", device_prop.major, device_prop.minor);
    printf("\nClock rate: %d KHz", device_prop.clockRate);
    printf("\nMultiprocessors Count: %d", device_prop.multiProcessorCount);
    printf("\nWarp size: %d threads", device_prop.warpSize);
    printf("\nTotal amount of Global Memory: %.2f KB", device_prop.totalGlobalMem / 1024.0);
    printf("\nTotal amount of Constant Memory: %.2f KB", device_prop.totalConstMem / 1024.0);
    printf("\nTotal amount of Shared Memory per Block: %.2f KB", device_prop.sharedMemPerBlock / 1024.0);
    printf("\nTotal amount of Shared Memory per Multiprocessor: %.2f KB", device_prop.sharedMemPerMultiprocessor / 1024.0);
    printf("\nMaximum size of each dimension of a Grid: (%d, %d, %d)", device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
    printf("\nMaximum size of each dimension of a Block: (%d, %d, %d)", device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
    printf("\nMaximum number of 32-bit registers per thread block: %d", device_prop.regsPerBlock);
    printf("\nMaximum number of threads per block: %d", device_prop.maxThreadsPerBlock);
    printf("\nMaximum number of threads per Multiprocessor: %d", device_prop.maxThreadsPerMultiProcessor);
    printf("\nMaximum number of warps per Multiprocessor: %d", device_prop.maxThreadsPerMultiProcessor / device_prop.warpSize);

    return 0;
}