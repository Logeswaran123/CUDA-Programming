#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "host_utils.h"
#include "cuda_utils.cuh"

#include <iostream>
#include <stdlib.h>

using namespace std;

/* //TODO
__global__ void ReduceSumGPU(int *input, int *sum, int size) {
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + tid;

	__syncthreads();
}
*/

template<unsigned int block_size>
__global__ void ReduceSumCompleteUnroll(int *input, int *temp, int size) {
	int tid = threadIdx.x;
	int block_offset = blockIdx.x * blockDim.x * 2;
	int index = block_offset + tid;

	int *i_data = input + block_offset;

	if ((index + blockDim.x) < size) {
		input[index] += input[index + blockDim.x];
	}
	__syncthreads();

	// Complete unrolling
	// Unrolling based on block size
	// In-place Reduction
	if (block_size >= 1024 && tid < 512)
		i_data[tid] += i_data[tid + 512];

	__syncthreads();

	if (block_size >= 512 && tid < 256)
		i_data[tid] += i_data[tid + 256];

	__syncthreads();

	if (block_size >= 256 && tid < 128)
		i_data[tid] += i_data[tid + 128];

	__syncthreads();

	if (block_size >= 128 && tid < 64)
		i_data[tid] += i_data[tid + 64];

	__syncthreads();

	// Warp unrolling
	if (tid < 32) {
		volatile int *vsmem = i_data;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	if (tid == 0) {
		temp[blockIdx.x] = i_data[0];
	}
}

template<unsigned int block_size>
__global__ void ReduceSumCompleteUnrollSMem(int *input, int *temp, int size) {
	__shared__ int smem[block_size];
	unsigned int tid = threadIdx.x;

	// Global index
	// 4 blocks of input data processed at a time
	int block_offset = blockIdx.x * blockDim.x * 4;  
	unsigned int index = block_offset + tid;  

	int temp_sum = 0;

	// Unrolling 4 blocks
	if (index + 3 * blockDim.x <= size) { // Boundary check
		int val1 = input[index + 0 * blockDim.x];
		int val2 = input[index + 1 * blockDim.x];
		int val3 = input[index + 2 * blockDim.x];
		int val4 = input[index + 3 * blockDim.x];
		temp_sum = val1 + val2 + val3 + val4;
	}

	smem[tid] = temp_sum;

	__syncthreads();

	// Complete unrolling
	// Unrolling based on block size
	// In-place Reduction in Shared Memory
	if (block_size >= 1024 && tid < 512) {
		smem[tid] += smem[tid + 512];
	}
	__syncthreads();

	if (block_size >= 512 && tid < 256) {
		smem[tid] += smem[tid + 256];
	}
	__syncthreads();

	if (block_size >= 256 && tid < 128) {
		smem[tid] += smem[tid + 128];
	}
	__syncthreads();

	if (block_size >= 128 && tid < 64) {
		smem[tid] += smem[tid + 64];
	}
	__syncthreads();

	// Warp unrolling
	if (tid < 32) {
		volatile int * vsmem = smem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	if (tid == 0) {
		temp[blockIdx.x] = smem[0];
	}
}

int main() {
	float gpu_total_milliseconds = 0;
	float kernel_milliseconds = 0;
	clock_t cpu_start, cpu_end;
	cudaEvent_t gpu_start, gpu_end, kernel_start, kernel_end;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_end);
	cudaEventCreate(&kernel_start);
	cudaEventCreate(&kernel_end);

	int size = 1 << 20; // 1MB
	int block_size = 1024;
	int byte_size = size * sizeof(int);

	int *host_input, *host_ref_unroll;
	host_input = (int*)malloc(byte_size);

	InitializeData(host_input, size, INIT_ONE);

	// Perform reduce sum on CPU
	cout << "\n-----CPU: Reduce Sum-----\n" << endl;
	cpu_start = clock();
	int cpu_result = ReductionSumCPU(host_input, size);
	cpu_end = clock();

	printf("CPU execution time (Function Only): %4.6f milliseconds",
		(double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC) * 1000.0);

	int gpu_result = 0;
	dim3 block(block_size);
	dim3 grid(ceil((size / block_size) / 2));

	printf("\nKernel launch parameters -> grid: (%d,%d,%d), block: (%d,%d,%d) \n",
            grid.x, grid.y, grid.z, block.x, block.y, block.z);

	int temp_array_byte_size = sizeof(int) * grid.x;
	host_ref_unroll = (int*)malloc(temp_array_byte_size);

	int *device_input, *device_temp;
	GPUErrorCheck(cudaEventRecord(gpu_start, 0));
	GPUErrorCheck(cudaMalloc((void**)&device_input, byte_size));
	GPUErrorCheck(cudaMemcpy(device_input, host_input, byte_size, cudaMemcpyHostToDevice));
	GPUErrorCheck(cudaMalloc((void**)&device_temp, temp_array_byte_size));
	GPUErrorCheck(cudaMemset(device_temp, 0, temp_array_byte_size));

	// Perform reduce sum with complete unroll on GPU
	cout << "\n-----Device: Reduce Sum with Complete Unroll-----\n" << endl;
	GPUErrorCheck(cudaEventRecord(kernel_start, 0));
	switch(block_size) {
		case 1024:
			ReduceSumCompleteUnroll<1024><<<grid, block>>>(device_input, device_temp, size);
			break;
		case 512:
			ReduceSumCompleteUnroll<512><<<grid, block>>>(device_input, device_temp, size);
			break;
		case 256:
			ReduceSumCompleteUnroll<256><<<grid, block>>>(device_input, device_temp, size);
			break;
		case 128:
			ReduceSumCompleteUnroll<128><<<grid, block>>>(device_input, device_temp, size);
			break;
	}
	GPUErrorCheck(cudaEventRecord(kernel_end, 0));
	GPUErrorCheck(cudaEventSynchronize(kernel_end));

	GPUErrorCheck(cudaDeviceSynchronize());
	GPUErrorCheck(cudaMemcpy(host_ref_unroll, device_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));

	for (int i = 0; i < grid.x; i++) {
		gpu_result += host_ref_unroll[i];
	}
	GPUErrorCheck(cudaEventRecord(gpu_end, 0));
	GPUErrorCheck(cudaEventSynchronize(gpu_end));
	GPUErrorCheck(cudaEventElapsedTime(&kernel_milliseconds, kernel_start, kernel_end));
	GPUErrorCheck(cudaEventElapsedTime(&gpu_total_milliseconds, gpu_start, gpu_end));

	CompareResults(gpu_result, cpu_result);

	printf("GPU Execution Time (Kernel Only): %4.6f milliseconds", kernel_milliseconds);
	printf("\nTotal GPU Execution Time (Malloc, Memcpy, Kernel): %4.6f milliseconds\n", gpu_total_milliseconds);

	GPUErrorCheck(cudaFree(device_temp));
	free(host_ref_unroll);

	grid.x = ceil((size / block_size) / 4);
	temp_array_byte_size = sizeof(int) * grid.x;
	host_ref_unroll = (int*)malloc(temp_array_byte_size);
	memset(host_ref_unroll, 0, temp_array_byte_size);
	GPUErrorCheck(cudaMalloc((void**)&device_temp, temp_array_byte_size));
	GPUErrorCheck(cudaMemset(device_temp, 0, temp_array_byte_size));

	cout << "\n-----Device: Reduce Sum using Shared Memory with Complete Unroll-----\n" << endl;
	GPUErrorCheck(cudaEventRecord(kernel_start, 0));
	switch(block_size) {
		case 1024:
			ReduceSumCompleteUnrollSMem<1024><<<grid, block>>>(device_input, device_temp, size);
			break;
		case 512:
			ReduceSumCompleteUnrollSMem<512><<<grid, block>>>(device_input, device_temp, size);
			break;
		case 256:
			ReduceSumCompleteUnrollSMem<256><<<grid, block>>>(device_input, device_temp, size);
			break;
		case 128:
			ReduceSumCompleteUnrollSMem<128><<<grid, block>>>(device_input, device_temp, size);
			break;
	}
	GPUErrorCheck(cudaEventRecord(kernel_end, 0));
	GPUErrorCheck(cudaEventSynchronize(kernel_end));
	GPUErrorCheck(cudaDeviceSynchronize());

	GPUErrorCheck(cudaMemcpy(host_ref_unroll, device_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));

	gpu_result = 0;
	for (int i = 0; i < grid.x; i++) {
		gpu_result += host_ref_unroll[i];
	}
	CompareResults(gpu_result, cpu_result);

	GPUErrorCheck(cudaEventElapsedTime(&kernel_milliseconds, kernel_start, kernel_end));
	printf("GPU Execution Time (Kernel Only): %4.6f milliseconds", kernel_milliseconds);

	GPUErrorCheck(cudaFree(device_temp));
	free(host_ref_unroll);

	GPUErrorCheck(cudaFree(device_input));
	free(host_input);

	GPUErrorCheck(cudaDeviceReset());
	return 0;
}