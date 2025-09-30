#include <iostream>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h> // 引入 OpenMP 头文件

// #define N 800
#define ITERATIONS 10

using namespace std;

double timestamp() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

// 优化后的函数
void yourFunction(float a, float b, float A[N][N], float B[N][N], float C[N][N]) {
    // 关键改动 1: 交换循环顺序为 i, k, j 以优化缓存访问
    // 关键改动 2: 将 b*C 的计算提前，以匹配 v0 的逻辑 C = (1+b)C + a*A*B
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] *= (1+b);
        }
    }

    // 使用 OpenMP 并行化最外层循环
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            // 将 A[i][k] 的值加载一次，供内层循环重复使用
            const float a_ik = A[i][k];
            for (int j = 0; j < N; j++) {
                // C[i][j] 的计算现在依赖于 A 的一行和 B 的一行
                // 内存访问是连续的，缓存命中率大大提高
                C[i][j] += a * a_ik * B[k][j];
            }
        }
    }
}

int main() {
    // 为了避免栈溢出，对于大矩阵使用动态内存分配
    float (*A)[N] = new float[N][N];
    float (*B)[N] = new float[N][N];
    float (*C)[N] = new float[N][N];
    float a = 0.5, b = 0.3;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)rand() / (float)(RAND_MAX / a);
            B[i][j] = (float)rand() / (float)(RAND_MAX / a);
            C[i][j] = 0;
        }
    }

    // 预热运行 (Warm-up)
    yourFunction(a, b, A, B, C);

    // 重置 C 矩阵以进行准确计时
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

    printf("N = %d\n", N);
    printf("GFLOPS/s = %lf\n", gflopsPerSecond);
    printf("GFLOPS = %lf\n", flops / 1000000000.0);
    printf("time(s) = %lf\n", time);

    // 释放动态分配的内存
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}