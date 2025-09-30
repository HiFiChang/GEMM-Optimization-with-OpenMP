#include <iostream>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

// #define N 1500
#define ITERATIONS 10
using namespace std;

double timestamp() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

void yourFunction(float a, float b, float* A, float* B, float* C) {
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            C[i * N + j] *= b;
            float tmp = 0;
            for (int k = 0; k < N; k++) {
                tmp += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] += tmp * a;
        }
    }
}

int main() {
    omp_set_num_threads(16);
    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(N * N * sizeof(float));
    float *C = (float*)malloc(N * N * sizeof(float));
    float a = 0.5, b = 0.3;

    // Initialize matrices A, B, and C
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (float)rand() / (float)(RAND_MAX / a);
            B[i * N + j] = (float)rand() / (float)(RAND_MAX / a);
            C[i * N + j] = 0;
        }
    }

    // Warm-up computation
    yourFunction(a, b, A, B, C);

    // Reset matrix C
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
        }
    }

    double time1 = timestamp();
    for (int numOfTimes = 0; numOfTimes < ITERATIONS; numOfTimes++) {
        yourFunction(a, b, A, B, C);
    }
    double time2 = timestamp();

    double time = (time2 - time1) / ITERATIONS;
    long long flops = 2 * N * N + 2 * N * N * (long long)N + 2 * N * N;
    double gflopsPerSecond = flops / (1000000000.0) / time;
    printf("N = %d\n", N);
    printf("GFLOPS/s = %lf\n", gflopsPerSecond);
    printf("GFLOPS = %lf\n", flops / (1000000000.0));
    printf("time(s) = %lf\n", time);

    free(A);
    free(B);
    free(C);

    return 0;
}
