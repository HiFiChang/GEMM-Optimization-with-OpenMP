#include <iostream>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define ITERATIONS 10
#define BLOCK_SIZE 32

using namespace std;

double timestamp() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

void yourFunction(float a, float b, float A[N][N], float B[N][N], float C[N][N]) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] *= (1+b);
        }
    }

    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                for (int i = bi; i < bi + BLOCK_SIZE; i++) {
                    for (int k = bk; k < bk + BLOCK_SIZE; k++) {
                        const float a_ik = a * A[i][k];
                        #pragma omp simd
                        for (int j = bj; j < bj + BLOCK_SIZE; j++) {
                            C[i][j] += a_ik * B[k][j];
                        }
                    }
                }
            }
        }
    }
}


int main() {
    float (*A)[N] = new float[N][N];
    float (*B)[N] = new float[N][N];
    float (*C)[N] = new float[N][N];
    float a = 0.5, b = 0.3;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)rand() / (float)(RAND_MAX / 1.0f);
            B[i][j] = (float)rand() / (float)(RAND_MAX / 1.0f);
            C[i][j] = 0;
        }
    }
    yourFunction(a, b, A, B, C);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
        }
    }

    double time1 = timestamp();
    for (int numOfTimes = 0; numOfTimes < ITERATIONS; numOfTimes++) {
        yourFunction(a, b, A, B, C);
    }
    double time2 = timestamp();

    double time = (time2 - time1) / ITERATIONS;
    
    double flops = 2.0 * N * N * N + 4.0 * N * N;
    double gflopsPerSecond = flops / (1000000000.0) / time;

    printf("N = %d, BLOCK_SIZE = %d\n", N, BLOCK_SIZE);
    printf("GFLOPS/s = %lf\n", gflopsPerSecond);
    printf("GFLOPS = %lf\n", flops / 1000000000.0);
    printf("time(s) = %lf\n", time);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}