#pragma once

#include <cstddef>
#include <cstdint>

#include <cutlass/cutlass.h>


template <typename T>
CUTLASS_HOST_DEVICE constexpr T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_gcd(T a, T b) {
    return b == 0 ? a : constexpr_gcd(b, a % b);
}

CUTLASS_HOST_DEVICE constexpr uint32_t next_power_of_two(uint32_t n) {
  if (n == 0) return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}
