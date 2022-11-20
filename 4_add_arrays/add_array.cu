#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.cuh"

#include <iostream>

__global__ void AddKernel(int *result, const int *arr1, const int *arr2, const int *arr3) {
    /** 
     * Add Three arrays.
    **/
    int tid = threadIdx.x;
    result[tid] = arr1[tid] + arr2[tid] + arr3[tid];
}

void HostAdd(int *result, const int arr1[], const int arr2[], const int arr3[], int arraysize) {
    for (int i = 0; i < arraysize; i++) {
        result[i] = arr1[i] + arr2[i] + arr3[i];
    }
}

int main() {
    float milliseconds = 0;
    clock_t host_start, host_end;
    cudaEvent_t kernel_start, kernel_end;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);

    const int arraysize = 5;
    const int a[arraysize] = { 0, 1, 2, 3, 4 };
    const int b[arraysize] = { 4, 3, 2, 1, 0 };
    const int c[arraysize] = { 1, 1, 1, 1, 1 };

    // Transfer data from host to device
    int *device_a, *device_b, *device_c;
    GPUErrorCheck(cudaMalloc((void**)&device_a, sizeof(int) * arraysize));
    GPUErrorCheck(cudaMalloc((void**)&device_b, sizeof(int) * arraysize));
    GPUErrorCheck(cudaMalloc((void**)&device_c, sizeof(int) * arraysize));
    GPUErrorCheck(cudaMemcpy(device_a, a, sizeof(int) * arraysize, cudaMemcpyHostToDevice));
    GPUErrorCheck(cudaMemcpy(device_b, b, sizeof(int) * arraysize, cudaMemcpyHostToDevice));
    GPUErrorCheck(cudaMemcpy(device_c, c, sizeof(int) * arraysize, cudaMemcpyHostToDevice));

    int* device_result;
    GPUErrorCheck(cudaMalloc((void**)&device_result, sizeof(int) * arraysize));

    // Set block and grid size
    int nx = 5;
    dim3 block(5);
    dim3 grid(nx / block.x);

    printf("\nAdding three arrays...\n\n");
    PrintInput(a, arraysize);
    printf(" + ");
    PrintInput(b, arraysize);
    printf(" + ");
    PrintInput(c, arraysize);

    // Device execution
    // Launch Add kernel
    GPUErrorCheck(cudaEventRecord(kernel_start, 0));
    AddKernel<<<grid, block>>>(device_result, device_a, device_b, device_c);
    GPUErrorCheck(cudaDeviceSynchronize());
    GPUErrorCheck(cudaEventRecord(kernel_end, 0));
    GPUErrorCheck(cudaEventSynchronize(kernel_end));

    // Transfer data from device to host
    int *host_result;
    host_result = (int*)malloc(sizeof(int) * arraysize);
    GPUErrorCheck(cudaMemcpy(host_result, device_result, sizeof(int) * arraysize, cudaMemcpyDeviceToHost));
    printf("\n\nGPU Result: ");
    PrintOutput(host_result, arraysize);

    GPUErrorCheck(cudaEventElapsedTime(&milliseconds, kernel_start, kernel_end));

    GPUErrorCheck(cudaFree(device_result));
    GPUErrorCheck(cudaFree(device_c));
    GPUErrorCheck(cudaFree(device_b));
    GPUErrorCheck(cudaFree(device_a));
    free(host_result);
    GPUErrorCheck(cudaDeviceReset());

    // CPU execution
    int *cpu_result;
    cpu_result = (int*)malloc(sizeof(int) * arraysize);

    host_start = clock();
    HostAdd(cpu_result, a, b, c, arraysize);
    host_end = clock();

    printf("\nCPU Result: ");
    PrintOutput(cpu_result, arraysize);
    free(cpu_result);

    printf("\n\nGPU Execution Time: %f milliseconds", milliseconds);
    printf("\nCPU Execution Time: %d - %d = %f", host_end, host_start, (double)(host_end - host_start) / CLOCKS_PER_SEC);

    return 0;
}