#ifndef HOST_UTILS_H
#define HOST_UTILS_H

#include <stdio.h>
#include <cstring>
#include <time.h>
#include <stdlib.h>
#include <sys/utime.h>
#include <fstream> 


enum INIT_PARAM {
	INIT_ZERO,
    INIT_ONE,
    INIT_RANDOM,
};

// Array Initialization
void InitializeData(int *input, const int array_size,
	INIT_PARAM PARAM = INIT_ONE, int x = 0);
void InitializeData(float *input, const int array_size,
	INIT_PARAM PARAM = INIT_ONE);

// Print an array
void PrintArray(int *input, const int array_size);
void PrintArray(float *input, const int array_size);

// Compare two values
void CompareResults(int a, int b);
void CompareResults(float a, float b);

// Compare two arrays
void CompareArrays(int *a, int *b, int size);
void CompareArrays(float *a, float *b, int size);

// Reduction operation in CPU
int ReductionSumCPU(int *input, const int size);
float ReductionSumCPU(float *input, const int size);

#endif // !HOST_UTILS_H