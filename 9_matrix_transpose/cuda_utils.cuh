#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#define GPUErrorCheck(status) { CudaErrorCheck(status, __FILE__, __LINE__); }

void CudaErrorCheck(cudaError_t status, const char *file, int line, bool abort = true) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Failed: %s, %s, %d", cudaGetErrorString(status), file, line);
        if (abort) {
            exit(status);
        }
    }
}