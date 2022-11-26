#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "host_utils.h"
#include "cuda_utils.cuh"

#include <iostream>
#include <stdlib.h>

using namespace std;

__global__ void Transpose1(int *matrix, int *transposed, int num_cols, int num_rows) {
    // Read row
    // Write column
    // Row major traversal in input matrix

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < num_cols && iy < num_rows) {
        transposed[ix * num_rows + iy] = matrix[iy * num_cols + ix];
    }
}

__global__ void Transpose2(int *matrix, int *transposed, int num_cols, int num_rows) {
    // Read column
    // Write row
    // Column major traversal in input matrix

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < num_cols && iy < num_rows) {
        transposed[iy * num_cols + ix] = matrix[ix * num_rows + iy];
    }
}

__global__ void Transpose1_Unroll(int *matrix, int *transposed, int num_cols, int num_rows) {
    // Read row
    // Write column
    // Row major traversal in input matrix
    // Unroll Loop

    int ix = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int ti = iy * num_cols + ix;
    int to = ix * num_rows + iy;

    if (ix + 3 * blockDim.x < num_cols && iy < num_rows) {
        transposed[to + num_rows * 0 * blockDim.x] = matrix[ti + 0 * blockDim.x];
        transposed[to + num_rows * 1 * blockDim.x] = matrix[ti + 1 * blockDim.x];
        transposed[to + num_rows * 2 * blockDim.x] = matrix[ti + 2 * blockDim.x];
        transposed[to + num_rows * 3 * blockDim.x] = matrix[ti + 3 * blockDim.x];
    }
}

__global__ void Transpose2_Unroll(int *matrix, int *transposed, int num_cols, int num_rows) {
    // Read column
    // Write row
    // Column major traversal in input matrix
    // Unroll Loop

    int ix = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int ti = iy * num_cols + ix;
    int to = ix * num_rows + iy;

    if (ix + 3 * blockDim.x < num_cols && iy < num_rows) {
        transposed[ti + 0 * blockDim.x] = matrix[to + 0 * blockDim.x*num_rows];
        transposed[ti + 1 * blockDim.x] = matrix[to + 1 * blockDim.x*num_rows];
        transposed[ti + 2 * blockDim.x] = matrix[to + 2 * blockDim.x*num_rows];
        transposed[ti + 3 * blockDim.x] = matrix[to + 3 * blockDim.x*num_rows];
    }
}

__global__ void TransposeDiagonal(int *matrix, int *transposed, int num_cols, int num_rows) {
    int blk_x = blockIdx.x;
    int blk_y = (blockIdx.x + blockIdx.y) % gridDim.x;

    int ix = blockIdx.x * blk_x + threadIdx.x;
    int iy = blockIdx.y * blk_y + threadIdx.y;

    if (ix < num_cols && iy < num_rows) {
        transposed[ix * num_rows + iy] = matrix[iy * num_cols + ix];
    }
}

int main() {
    cout << "\n---------------------Matrix Transpose---------------------\n" << endl;
    clock_t cpu_start, cpu_end;

    int num_rows = 1024;
    int num_cols = 1024;
    int size = num_rows * num_cols; // 1MB
    int byte_size = size * sizeof(int);

    int *host_input, *host_transposed, *device_result;
    host_input = (int*)malloc(byte_size);
    host_transposed = (int*)malloc(byte_size);

    InitializeData(host_input, size, INIT_RANGE);

    // Perform matrix transpose on CPU
    cout << "\n-----Transpose on CPU-----" << endl;
    cpu_start = clock();
    MatrixTranposeCPU(host_input, host_transposed, num_cols, num_rows);
    cpu_end = clock();

    printf("\nCPU execution time (Function Only): %4.6f milliseconds\n",
        (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC) * 1000.0);

    /***************************************************************************/
    float kernel_milliseconds = 0;
    int block_x = 128;
    int block_y = 8;
    cudaEvent_t kernel_start, kernel_end;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);

    int *device_matrix, *device_transposed;
    GPUErrorCheck(cudaMalloc((void**)&device_matrix, byte_size));
    GPUErrorCheck(cudaMemcpy(device_matrix, host_input, byte_size, cudaMemcpyHostToDevice));

    dim3 blocks(block_x, block_y);
    dim3 grid(num_cols / block_x + 1, num_rows / block_y + 1);

    // Transpose: Read Row, Write Column
    cout << "\n-----Transpose on Device: Read Row, Write Column-----" << endl;
    GPUErrorCheck(cudaMalloc((void**)&device_transposed, byte_size));
    GPUErrorCheck(cudaEventRecord(kernel_start, 0));
    Transpose1<<<grid, blocks>>>(device_matrix, device_transposed, num_cols, num_rows);
    GPUErrorCheck(cudaEventRecord(kernel_end, 0));
    GPUErrorCheck(cudaEventSynchronize(kernel_end));
    GPUErrorCheck(cudaDeviceSynchronize());

    device_result = (int*)malloc(byte_size);
    GPUErrorCheck(cudaMemcpy(device_result, device_transposed, byte_size, cudaMemcpyDeviceToHost));
    CompareArrays(host_transposed, device_result, size);
    GPUErrorCheck(cudaFree(device_transposed));
    free(device_result);

    GPUErrorCheck(cudaEventElapsedTime(&kernel_milliseconds, kernel_start, kernel_end));
    printf("GPU Execution Time (Kernel Only): %4.6f milliseconds\n", kernel_milliseconds);

    // Transpose: Read Column, Write Row
    cout << "\n-----Transpose on Device: Read Column, Write Row-----" << endl;
    GPUErrorCheck(cudaMalloc((void**)&device_transposed, byte_size));
    GPUErrorCheck(cudaEventRecord(kernel_start, 0));
    Transpose2<<<grid, blocks>>>(device_matrix, device_transposed, num_cols, num_rows);
    GPUErrorCheck(cudaEventRecord(kernel_end, 0));
    GPUErrorCheck(cudaEventSynchronize(kernel_end));
    GPUErrorCheck(cudaDeviceSynchronize());

    device_result = (int*)malloc(byte_size);
    GPUErrorCheck(cudaMemcpy(device_result, device_transposed, byte_size, cudaMemcpyDeviceToHost));
    CompareArrays(host_transposed, device_result, size);
    GPUErrorCheck(cudaFree(device_transposed));
    free(device_result);

    GPUErrorCheck(cudaEventElapsedTime(&kernel_milliseconds, kernel_start, kernel_end));
    printf("GPU Execution Time (Kernel Only): %4.6f milliseconds\n", kernel_milliseconds);

    // Transpose: Read Row, Write Column, with Unrolled loop
    cout << "\n-----Transpose on Device: Read Row, Write Column, with Unrolled loop-----" << endl;
    GPUErrorCheck(cudaMalloc((void**)&device_transposed, byte_size));
    GPUErrorCheck(cudaEventRecord(kernel_start, 0));
    Transpose1_Unroll<<<grid, blocks>>>(device_matrix, device_transposed, num_cols, num_rows);
    GPUErrorCheck(cudaEventRecord(kernel_end, 0));
    GPUErrorCheck(cudaEventSynchronize(kernel_end));
    GPUErrorCheck(cudaDeviceSynchronize());

    device_result = (int*)malloc(byte_size);
    GPUErrorCheck(cudaMemcpy(device_result, device_transposed, byte_size, cudaMemcpyDeviceToHost));
    CompareArrays(host_transposed, device_result, size);
    GPUErrorCheck(cudaFree(device_transposed));
    free(device_result);

    GPUErrorCheck(cudaEventElapsedTime(&kernel_milliseconds, kernel_start, kernel_end));
    printf("GPU Execution Time (Kernel Only): %4.6f milliseconds\n", kernel_milliseconds);

    // Transpose: Read Column, Write Row, with Unrolled loop
    cout << "\n-----Transpose on Device: Read Column, Write Row, with Unrolled loop-----" << endl;
    GPUErrorCheck(cudaMalloc((void**)&device_transposed, byte_size));
    GPUErrorCheck(cudaEventRecord(kernel_start, 0));
    Transpose2_Unroll<<<grid, blocks>>>(device_matrix, device_transposed, num_cols, num_rows);
    GPUErrorCheck(cudaEventRecord(kernel_end, 0));
    GPUErrorCheck(cudaEventSynchronize(kernel_end));
    GPUErrorCheck(cudaDeviceSynchronize());

    device_result = (int*)malloc(byte_size);
    GPUErrorCheck(cudaMemcpy(device_result, device_transposed, byte_size, cudaMemcpyDeviceToHost));
    CompareArrays(host_transposed, device_result, size);
    GPUErrorCheck(cudaFree(device_transposed));
    free(device_result);

    GPUErrorCheck(cudaEventElapsedTime(&kernel_milliseconds, kernel_start, kernel_end));
    printf("GPU Execution Time (Kernel Only): %4.6f milliseconds\n", kernel_milliseconds);

    // // Transpose: Diagonal method
    // cout << "\n-----Transpose on Device: Diagonal method-----" << endl;
    // GPUErrorCheck(cudaMalloc((void**)&device_transposed, byte_size));
    // GPUErrorCheck(cudaEventRecord(kernel_start, 0));
    // TransposeDiagonal<<<grid, blocks>>>(device_matrix, device_transposed, num_cols, num_rows);
    // GPUErrorCheck(cudaEventRecord(kernel_end, 0));
    // GPUErrorCheck(cudaEventSynchronize(kernel_end));
    // GPUErrorCheck(cudaDeviceSynchronize());

    // device_result = (int*)malloc(byte_size);
    // GPUErrorCheck(cudaMemcpy(device_result, device_transposed, byte_size, cudaMemcpyDeviceToHost));
    // CompareArrays(host_transposed, device_result, size);
    // GPUErrorCheck(cudaFree(device_transposed));
    // free(device_result);

    // GPUErrorCheck(cudaEventElapsedTime(&kernel_milliseconds, kernel_start, kernel_end));
    // printf("GPU Execution Time (Kernel Only): %4.6f milliseconds\n", kernel_milliseconds);

    GPUErrorCheck(cudaFree(device_matrix));
    free(host_transposed);
    free(host_input);

    GPUErrorCheck(cudaDeviceReset());
    return 0;
}