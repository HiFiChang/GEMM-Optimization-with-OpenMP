#include <iostream>
#include <vector>
#include <omp.h>
// #include <mkl.h>

// #define N 1500
#define ITERATIONS 10
#define BLOCK_SIZE 16

alignas(64) float localA[BLOCK_SIZE][BLOCK_SIZE];
alignas(64) float localB[BLOCK_SIZE][BLOCK_SIZE];
alignas(64) float localC[BLOCK_SIZE][BLOCK_SIZE];
#pragma omp threadprivate(localA, localB, localC)

void gemm(const std::vector<float> &matA, const std::vector<float> &matB, std::vector<float> &matC, size_t matSize) {
    // Transpose matrix B
    std::vector<float> matBTrans(matB.size());
    // mkl_somatcopy('R', 'T', matSize, matSize, 1.0, matB.data(), matSize, matBTrans.data(), matSize);

    size_t blockNum = matSize / BLOCK_SIZE;

    // Clear matC
    #pragma omp parallel for
    for (size_t i = 0; i < matC.size(); i++) {
        matC[i] = 0;
    }

    // Traverse blocks
    #pragma omp parallel for collapse(2)
    for (size_t bi = 0; bi < blockNum; bi++) {
        for (size_t bj = 0; bj < blockNum; bj++) {
            for (size_t bk = 0; bk < blockNum; bk++) {
                // Copy local block
                for (size_t i = 0; i < BLOCK_SIZE; i++) {
                    for (size_t j = 0; j < BLOCK_SIZE; j++) {
                        size_t aIdx = bi * BLOCK_SIZE * matSize + i * matSize + bk * BLOCK_SIZE + j;
                        size_t bIdx = bj * BLOCK_SIZE * matSize + i * matSize + bk * BLOCK_SIZE + i;
                        localA[i][j] = matA[aIdx];
                        localB[i][j] = matBTrans[bIdx];
                        localC[i][j] = 0;
                    }
                }

                // Block GEMM
                for (size_t i = 0; i < BLOCK_SIZE; i++) {
                    for (size_t j = 0; j < BLOCK_SIZE; j++) {
                        #pragma omp simd
                        for (size_t k = 0; k < BLOCK_SIZE; k++) {
                            localC[i][j] += localA[i][k] * localB[j][k];
                        }
                    }
                }

                // Copy localC back to matC
                for (size_t i = 0; i < BLOCK_SIZE; i++) {
                    for (size_t j = 0; j < BLOCK_SIZE; j++) {
                        size_t cIdx = bi * BLOCK_SIZE * matSize + i * matSize + bj * BLOCK_SIZE + j;
                        #pragma omp atomic
                        matC[cIdx] += localC[i][j];
                    }
                }
            }
        }
    }
}

int main() {
    omp_set_num_threads(16);
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N);
    float a = 0.5, b = 0.3;

    // Initialize matrices A and B
    for (size_t i = 0; i < N * N; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Perform warm-up computation
    gemm(A, B, C, N);

    // Reset matrix C
    std::fill(C.begin(), C.end(), 0.0f);

    double time1 = omp_get_wtime();
    for (int numOfTimes = 0; numOfTimes < ITERATIONS; numOfTimes++) {
        gemm(A, B, C, N);
    }
    double time2 = omp_get_wtime();

    double time = (time2 - time1) / ITERATIONS;
    long long flops = 2 * N * N + 2 * N * N * static_cast<long long>(N) + 2 * N * N;
    double gflopsPerSecond = flops / 1e9 / time;
    printf("N = %d\n", N);
    // printf("Block Size = %d\n", BLOCK_SIZE);
    printf("GFLOPS/s = %lf\n", gflopsPerSecond);
    printf("GFLOPS = %lf\n", flops / 1e9);
    printf("time(s) = %lf\n", time);
    
    return 0;
}
