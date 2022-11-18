#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

__global__ void AddKernel(int *c, const int *a, const int *b) {
    /** 
     * Add two arrays.
    **/
    int tid = threadIdx.x;
    c[tid] = a[tid] + b[tid];
}

void PrintInput(const int arr[], const int arraysize) {
    printf("[ ");
    for (int i = 0; i < arraysize; i++) {
        printf("%d, ", arr[i]);
    }
    printf("]");
}

void PrintOutput(int *arr, const int arraysize) {
    printf("[ ");
    for (int i = 0; i < arraysize; i++) {
        printf("%d, ", arr[i]);
    }
    printf("]");
}

int main() {
    const int arraysize = 5;
    const int a[arraysize] = { 0, 1, 2, 3, 4 };
    const int b[arraysize] = { 4, 3, 2, 1, 0 };

    // Transfer data from host to device
    int *device_a;
    cudaMalloc((void**)&device_a, sizeof(int) * arraysize);
    cudaMemcpy(device_a, a, sizeof(int) * arraysize, cudaMemcpyHostToDevice);

    int *device_b;
    cudaMalloc((void**)&device_b, sizeof(int) * arraysize);
    cudaMemcpy(device_b, b, sizeof(int) * arraysize, cudaMemcpyHostToDevice);

    int* device_c;
    cudaMalloc((void**)&device_c, sizeof(int) * arraysize);

    // Set block and grid size
    int nx = 5;
    dim3 block(5);
    dim3 grid(nx / block.x);

    // Launch Add kernel
    printf("\nAdding two vectors...\n\n");
    PrintInput(a, arraysize);
    printf("\t + \t");
    PrintInput(b, arraysize);
    AddKernel<<<grid, block>>>(device_c, device_a, device_b);
    cudaDeviceSynchronize();

    int *c;
    c = (int*)malloc(sizeof(int) * arraysize);
    cudaMemcpy(c, device_c, sizeof(int) * arraysize, cudaMemcpyDeviceToHost);
    printf("\n\nResult: ");
    PrintOutput(c, arraysize);

    cudaFree(device_c);
    free(c);    
    cudaDeviceReset();
    return 0;
}