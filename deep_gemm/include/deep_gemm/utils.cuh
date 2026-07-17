#pragma once

#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <iomanip>
#include "utils_rtc.cuh"

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

#define CHECK_HGGC(call) \
{ \
    hggcError_t err = call; \
    if (err != hggcSuccess) { \
        printf("%s:%d HGGC API error: %s\n", __FILE__, __LINE__, hggcGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
