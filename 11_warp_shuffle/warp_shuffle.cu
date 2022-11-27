#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "host_utils.h"
#include "cuda_utils.cuh"

#include <iostream>
#include <stdlib.h>

using namespace std;

#define MASK 0xffffffff
#define WARP_SIZE 32

__global__ void WarpShuffle(int *input, int *output, int srcLane, int width=WARP_SIZE) {
    auto in = input[threadIdx.x];
    auto out = __shfl_sync(MASK, in, srcLane, width);
    output[threadIdx.x] = out;
}

__global__ void WarpShuffleUP(int *input, int *output, unsigned int delta, int width=WARP_SIZE) {
    auto in = input[threadIdx.x];
    auto out = __shfl_up_sync(MASK, in, delta, width);
    output[threadIdx.x] = out;
}

__global__ void WarpShuffleDOWN(int *input, int *output, unsigned int delta, int width=WARP_SIZE) {
    auto in = input[threadIdx.x];
    auto out = __shfl_down_sync(MASK, in, delta, width);
    output[threadIdx.x] = out;
}

__global__ void WarpShuffleXOR(int *input, int *output, int laneMask, int width=WARP_SIZE) {
    auto in = input[threadIdx.x];
    auto out = __shfl_xor_sync(MASK, in, laneMask, width);
    output[threadIdx.x] = out;
}

int main() {
    int srcLane, delta, laneMask;
    int size = 64;
    int byte_size = size * sizeof(int);

    int *host_input, *host_output;
    host_input = (int*)malloc(byte_size);
    InitializeData(host_input, size, INIT_RANGE);
    cout << "\nInput Array:\n" << endl;
    PrintArray(host_input, size);

    auto nx = size;
    dim3 block(nx);
    dim3 grid(nx / block.x);

    int *device_input, *device_output;
    GPUErrorCheck(cudaMalloc((void**)&device_input, byte_size));
    GPUErrorCheck(cudaMemcpy(device_input, host_input, byte_size, cudaMemcpyHostToDevice));

    cout << "\nWarp Shuffle:\n" << endl;
    srcLane = 4;
    GPUErrorCheck(cudaMalloc((void**)&device_output, byte_size));
    WarpShuffle<<<grid, block>>>(device_input, device_output, srcLane);
    GPUErrorCheck(cudaDeviceSynchronize());
    host_output = (int*)malloc(byte_size);
    GPUErrorCheck(cudaMemcpy(host_output, device_output, byte_size, cudaMemcpyDeviceToHost));
    PrintArray(host_output, size);
    GPUErrorCheck(cudaFree(device_output));
    free(host_output);

    cout << "\nWarp Shuffle Up:\n" << endl;
    delta = 4;
    GPUErrorCheck(cudaMalloc((void**)&device_output, byte_size));
    WarpShuffleUP<<<grid, block>>>(device_input, device_output, delta);
    GPUErrorCheck(cudaDeviceSynchronize());
    host_output = (int*)malloc(byte_size);
    GPUErrorCheck(cudaMemcpy(host_output, device_output, byte_size, cudaMemcpyDeviceToHost));
    PrintArray(host_output, size);
    GPUErrorCheck(cudaFree(device_output));
    free(host_output);

    cout << "\nWarp Shuffle Down:\n" << endl;
    delta = 4;
    GPUErrorCheck(cudaMalloc((void**)&device_output, byte_size));
    WarpShuffleDOWN<<<grid, block>>>(device_input, device_output, delta);
    GPUErrorCheck(cudaDeviceSynchronize());
    host_output = (int*)malloc(byte_size);
    GPUErrorCheck(cudaMemcpy(host_output, device_output, byte_size, cudaMemcpyDeviceToHost));
    PrintArray(host_output, size);
    GPUErrorCheck(cudaFree(device_output));
    free(host_output);

    cout << "\nWarp Shuffle XOR:\n" << endl;
    laneMask = 1;
    GPUErrorCheck(cudaMalloc((void**)&device_output, byte_size));
    WarpShuffleXOR<<<grid, block>>>(device_input, device_output, laneMask);
    GPUErrorCheck(cudaDeviceSynchronize());
    host_output = (int*)malloc(byte_size);
    GPUErrorCheck(cudaMemcpy(host_output, device_output, byte_size, cudaMemcpyDeviceToHost));
    PrintArray(host_output, size);
    GPUErrorCheck(cudaFree(device_output));
    free(host_output);

    GPUErrorCheck(cudaFree(device_input));
    free(host_input);

    GPUErrorCheck(cudaDeviceReset());
    return 0;
}