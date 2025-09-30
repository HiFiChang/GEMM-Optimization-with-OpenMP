#include <iostream>
// #include <fstream>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

// #define N 1500
#define ITERATIONS 10
#define BLOCK_SIZE 64

using namespace std;

double timestamp() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

// Transpose a block of the matrix B into BT to improve cache locality
void transposeBlock(float B[N][N], float BT[N][N], int startRow, int startCol) {
    #pragma omp parallel for collapse(2) // Parallelize the transposition
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            if (startRow + i < N && startCol + j < N) {
                BT[startCol + j][startRow + i] = B[startRow + i][startCol + j];
            }
        }
    }
}

// Transpose the entire matrix B into BT in blocks
void transpose(float B[N][N], float BT[N][N]) {
    #pragma omp parallel for collapse(2) // Parallelize the transposition
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            transposeBlock(B, BT, i, j);
        }
    }
}

void yourFunction(float a, float b, float A[N][N], float BT[N][N], float C[N][N]) {
    alignas(64) float localA[BLOCK_SIZE][BLOCK_SIZE]; // Allocate local blocks
    alignas(64) float localB[BLOCK_SIZE][BLOCK_SIZE];
    alignas(64) float localC[BLOCK_SIZE][BLOCK_SIZE];
    #pragma omp parallel for private(localA, localB, localC) // Parallelize the computation

    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                // Copy local blocks
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    for (int j = 0; j < BLOCK_SIZE; j++) {
                        if (bi + i < N && bk + j < N) {
                            localA[i][j] = A[bi + i][bk + j];
                        } else {
                            localA[i][j] = 0.0f;
                        }
                        if (bk + i < N && bj + j < N) {
                            localB[i][j] = BT[bj + j][bk + i]; // Using transposed B
                        } else {
                            localB[i][j] = 0.0f;
                        }
                        localC[i][j] = 0.0f;
                    }
                }

                // Perform block GEMM
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    // for (int j = 0; j < BLOCK_SIZE; j++) {
                    for (int k = 0; k < BLOCK_SIZE; k++) {
                        #pragma omp simd // SIMD optimization for inner loop
                        // for (int k = 0; k < BLOCK_SIZE; k++) {
                        for (int j = 0; j < BLOCK_SIZE; j++) {    
                            localC[i][j] += localA[i][k] * localB[k][j];
                            // localC[i][j] += localA[i][k] * localB[j][k];
                        }
                    }
                }

                // Copy the result back to matrix C
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    for (int j = 0; j < BLOCK_SIZE; j++) {
                        if (bi + i < N && bj + j < N) {
                            C[bi + i][bj + j] += localC[i][j] * a;
                        }
                    }
                }
            }
        }
    }

    // Scale matrix C by b
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] *= b;
        }
    }
}

int main() {
    omp_set_num_threads(16); // Set number of threads for OpenMP
    float (*A)[N] = new float[N][N]; // Dynamic memory allocation for large matrices
    float (*B)[N] = new float[N][N];
    float (*BT)[N] = new float[N][N];
    float (*C)[N] = new float[N][N];
    float a = 0.5, b = 0.3;

    // Initialize matrices A, B, and C
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)rand() / (float)(RAND_MAX / a);
            B[i][j] = (float)rand() / (float)(RAND_MAX / a);
            C[i][j] = 0;
        }
    }

    // Transpose matrix B for better cache performance during multiplication
    transpose(B, BT);

    // Perform warm-up computation
    yourFunction(a, b, A, BT, C);

    // Reset matrix C
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
        }
    }

    double time1 = timestamp();
    for (int numOfTimes = 0; numOfTimes < ITERATIONS; numOfTimes++) {
        yourFunction(a, b, A, BT, C);
    }
    double time2 = timestamp();

    double time = (time2 - time1) / ITERATIONS;
    long long flops = 2 * N * N + 2 * N * N * (long long)N + 2 * N * N;
    double gflopsPerSecond = flops / (1000000000.0) / time;
    printf("N = %d\n", N);
    printf("GFLOPS/s = %lf\n", gflopsPerSecond);
    printf("GFLOPS = %lf\n", flops / (1000000000.0));
    printf("time(s) = %lf\n", time);

    delete[] A;
    delete[] B;
    delete[] BT;
    delete[] C;

    return 0;
}