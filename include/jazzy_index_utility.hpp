#pragma once

#include <type_traits>

namespace jazzy::detail {

template <typename T>
constexpr bool IS_STRICTLY_ARITHMETIC_V =
    std::is_integral_v<T> || std::is_floating_point_v<T>;

template <typename T>
[[nodiscard]] constexpr T clamp_value(T value, T lo, T hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

}  // namespace jazzy::detail
