#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

__global__ void CudaHello(){
    /** 
     * Print "Hello World from GPU!" string from GPU scope.
    **/
    printf("Hello World from GPU!\n");
}

int main() {
    dim3 block(4);
    dim3 grid(8);

    CudaHello<<<grid, block>>>();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}