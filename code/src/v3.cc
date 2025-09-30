#include <iostream>
#include <sys/time.h>
#include <stdlib.h> // posix_memalign 和 free 函数需要此头文件
#include <stdio.h>
#include <omp.h>
#include <malloc.h> // 在某些Linux发行版中，posix_memalign可能在这里

// #define N 800
#define ITERATIONS 10
#define BLOCK_SIZE 32
// 定义内存对齐边界，64字节是现代CPU缓存行的通用大小
#define ALIGNMENT 64

using namespace std;

double timestamp() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

// 分块优化后的函数 (此函数无需改动)
void yourFunction(float a, float b, float A[N][N], float B[N][N], float C[N][N]) {
    // 关键改动: 添加 (1+b)*C 的计算，与 v0 逻辑匹配
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
    // --- 关键改动：使用posix_memalign进行内存对齐分配 ---

    // 1. 先声明原始指针和二维数组指针
    float *A_ptr, *B_ptr, *C_ptr;
    
    // 2. 使用posix_memalign分配对齐的内存块
    //    函数原型: int posix_memalign(void **memptr, size_t alignment, size_t size);
    if (posix_memalign((void**)&A_ptr, ALIGNMENT, N * N * sizeof(float)) != 0) {
        printf("Memory allocation failed for A\n");
        return 1;
    }
    if (posix_memalign((void**)&B_ptr, ALIGNMENT, N * N * sizeof(float)) != 0) {
        printf("Memory allocation failed for B\n");
        free(A_ptr); // 如果B分配失败，也要释放已分配的A
        return 1;
    }
    if (posix_memalign((void**)&C_ptr, ALIGNMENT, N * N * sizeof(float)) != 0) {
        printf("Memory allocation failed for C\n");
        free(A_ptr);
        free(B_ptr);
        return 1;
    }

    // 3. 将分配好的1D指针强制转换为2D数组指针，以便后续代码无缝使用
    float (*A)[N] = (float (*)[N])A_ptr;
    float (*B)[N] = (float (*)[N])B_ptr;
    float (*C)[N] = (float (*)[N])C_ptr;

    float a = 0.5, b = 0.3;

    // 初始化
    #pragma omp parallel for
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
    #pragma omp parallel for
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

    printf("N = %d, BLOCK_SIZE = %d, ALIGNMENT = %d\n", N, BLOCK_SIZE, ALIGNMENT);
    printf("GFLOPS/s = %lf\n", gflopsPerSecond);
    printf("GFLOPS = %lf\n", flops / 1000000000.0);
    printf("time(s) = %lf\n", time);

    // --- 释放内存 ---
    free(A_ptr);
    free(B_ptr);
    free(C_ptr);

    return 0;
}