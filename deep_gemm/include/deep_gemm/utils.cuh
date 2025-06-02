#pragma once

#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <iomanip>

#ifdef __CLION_IDE__
__host__ __device__ __forceinline__ void host_device_printf(const char* format, ...) { asm volatile("trap;"); }
#define printf host_device_printf
#endif

class AssertionException : public std::exception {
private:
    std::string message{};

public:
    explicit AssertionException(const std::string& message) : message(message) {}

    const char *what() const noexcept override { return message.c_str(); }
};

#ifndef DG_HOST_ASSERT
#define DG_HOST_ASSERT(cond)                                        \
do {                                                                \
    if (not (cond)) {                                               \
        printf("Assertion failed: %s:%d, condition: %s\n",          \
               __FILE__, __LINE__, #cond);                          \
        throw AssertionException("Assertion failed: " #cond);       \
    }                                                               \
} while (0)
#endif

#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond)                                                          \
do {                                                                                    \
    if (not (cond)) {                                                                   \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond);  \
        asm("trap;");                                                                   \
    }                                                                                   \
} while (0)
#endif

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

template <typename T>
__device__ __host__ constexpr T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_gcd(T a, T b) {
    return b == 0 ? a : constexpr_gcd(b, a % b);
}

#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("%s:%d CUDA API error: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

template<typename T>
void print_to_file(const T*           result,
                   const int          size,
                   const char*        file,
                   cudaStream_t       stream    = 0,
                   std::ios::openmode open_mode = std::ios::out);

template<typename T>
void print_to_file(
    const T* result, const int size, const char* file, cudaStream_t stream, std::ios::openmode open_mode) {
    cudaDeviceSynchronize();
    
    CHECK_CUDA(cudaGetLastError());
    printf("[INFO] file: %s with size %d.\n", file, size);
    std::ofstream outFile(file, open_mode);
    if (outFile) {
        T* tmp = new T[size];
        CHECK_CUDA(cudaMemcpyAsync(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost, stream));
        if constexpr(std::is_integral<T>::value) {
            outFile << std::fixed << std::setprecision(0);
        }
        for (int i = 0; i < size; ++i) {
            float val = (float)(tmp[i]);
            outFile << val << std::endl;
        }
        delete[] tmp;
    } else {
        throw std::runtime_error(std::string("[ERROR] Cannot open file: ") + file + "\n");
    }
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
}

template void
print_to_file(const int* result, const int size, const char* file, cudaStream_t stream, std::ios::openmode open_mode);