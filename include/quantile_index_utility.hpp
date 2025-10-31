#pragma once

#include <type_traits>

namespace bucket_index {

namespace detail {

template <typename T>
constexpr bool is_strictly_arithmetic_v =
    std::is_integral_v<T> || std::is_floating_point_v<T>;

template <typename T>
[[nodiscard]] constexpr T clamp_value(T value, T lo, T hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

}  // namespace detail

}  // namespace bucket_index

