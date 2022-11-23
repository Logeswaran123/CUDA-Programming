#include "host_utils.h"

void InitializeData(int *input, const int array_size, INIT_PARAM PARAM, int x) {
	if (PARAM == INIT_ONE) {
		for (int i = 0; i < array_size; i++) {
			input[i] = 1;
		}
	}
	else if (PARAM == INIT_RANDOM) {
		time_t t;
		srand((unsigned)time(&t));
		for (int i = 0; i < array_size; i++) {
			input[i] = (int)(rand() & 0xFF);
		}
	}
}

void InitializeData(float *input, const int array_size, INIT_PARAM PARAM) {
	if (PARAM == INIT_ONE) {
		for (int i = 0; i < array_size; i++) {
			input[i] = 1;
		}
	}
	else if (PARAM == INIT_RANDOM) {
		srand(time(NULL));
		for (int i = 0; i < array_size; i++) {
			input[i] = rand() % 10;
		}
	}
}

void PrintArray(int *input, const int array_size) {
	for (int i = 0; i < array_size; i++) {
		if (!(i == (array_size - 1))) {
			printf("%d,", input[i]);
		}
		else {
			printf("%d \n", input[i]);
		}
	}
}

void PrintArray(float *input, const int array_size) {
	for (int i = 0; i < array_size; i++) {
		if (!(i == (array_size - 1))) {
			printf("%f,", input[i]);
		}
		else {
			printf("%f \n", input[i]);
		}
	}
}

void CompareResults(int lhs, int rhs) {
	printf("LHS: %d, RHS: %d \n", lhs, rhs);

	if (lhs == rhs) {
		printf("LHS and RHS values are equal.\n\n");
		return;
	}
	printf("LHS and RHS values are different.\n\n");
}

void CompareResults(float lhs, float rhs) {
	printf("LHS: %f, RHS: %f \n", lhs, rhs);

	if (lhs == rhs) {
		printf("LHS and RHS values are equal.\n\n");
		return;
	}
	printf("LHS and RHS values are different.\n\n");
}

void CompareArrays(int *a, int *b, int size) {
    bool flag = true;
	for (int  i = 0; i < size; i++) {
		if (a[i] != b[i]) {
			flag = false;
			printf("Arrays are different: %d vs. %d at index %d\n\n", a[i], b[i], i);
		}
	}
    if (flag) { printf("\nArrays are equal.\n\n"); }
    else { printf("\nArrays are different.\n\n"); }
}

void CompareArrays(float *a, float *b, int size) {
    bool flag = true;
	for (int  i = 0; i < size; i++) {
		if (a[i] != b[i]) {
			flag = false;
			printf("Arrays are different: %f vs. %f at index %d\n\n", a[i], b[i], i);
		}
	}
    if (flag) { printf("\nArrays are equal.\n\n"); }
    else { printf("\nArrays are different.\n\n"); }
}

int ReductionSumCPU(int *input, const int size) {
	int sum = 0;
	for (int i = 0; i < size; i++) {
		sum += input[i];
	}
	return sum;
}

float ReductionSumCPU(float *input, const int size) {
	float sum = 0;
	for (int i = 0; i < size; i++) {
		sum += input[i];
	}
	return sum;
}