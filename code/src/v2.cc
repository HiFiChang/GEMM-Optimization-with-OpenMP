#include <iostream>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

// #define N 800
#define ITERATIONS 10
// 定义块的大小，这是一个可调参数。通常选择32或64。
// 理想情况下，3个块（A的一个块，B的一个块，C的一个块）能装入L1或L2缓存。
// 3 * (BLOCK_SIZE * BLOCK_SIZE * sizeof(float)) << L2_CACHE_SIZE
#define BLOCK_SIZE 32

using namespace std;

double timestamp() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

// 分块优化后的函数
void yourFunction(float a, float b, float A[N][N], float B[N][N], float C[N][N]) {
    // 关键改动: 添加 (1+b)*C 的计算，与 v0 逻辑匹配
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] *= (1+b);
        }
    }

    // 引入了三层新的外层循环 bi, bj, bk 来遍历矩阵块
    #pragma omp parallel for collapse(2) // 在块级别上进行并行化，使用collapse(2)让线程在bi和bj上分配
    for (int bi = 0; bi < N; bi += BLOCK_SIZE) {
        for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
            // 注意：bk循环不能并行化，因为它涉及到对同一C块的累加
            for (int bk = 0; bk < N; bk += BLOCK_SIZE) {
                // 在块内部，我们仍然使用缓存友好的 i, k, j 循环顺序
                for (int i = bi; i < bi + BLOCK_SIZE; i++) {
                    for (int k = bk; k < bk + BLOCK_SIZE; k++) {
                        const float a_ik = a * A[i][k];
                        // 建议编译器对最内层循环进行SIMD向量化
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
    // 使用动态内存分配
    float (*A)[N] = new float[N][N];
    float (*B)[N] = new float[N][N];
    float (*C)[N] = new float[N][N];
    float a = 0.5, b = 0.3;

    // 初始化
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)rand() / (float)(RAND_MAX / 1.0f);
            B[i][j] = (float)rand() / (float)(RAND_MAX / 1.0f);
            C[i][j] = 0;
        }
    }

    // 预热运行 (Warm-up)
    yourFunction(a, b, A, B, C);

    // 重置 C 矩阵
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
    
    // 关键改动: FLOPs 计算与 v0 保持一致
    double flops = 2.0 * N * N * N + 4.0 * N * N;
    double gflopsPerSecond = flops / (1000000000.0) / time;

    printf("N = %d, BLOCK_SIZE = %d\n", N, BLOCK_SIZE);
    printf("GFLOPS/s = %lf\n", gflopsPerSecond);
    printf("GFLOPS = %lf\n", flops / 1000000000.0);
    printf("time(s) = %lf\n", time);

    // 释放内存
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}