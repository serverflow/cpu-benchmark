#pragma once

#if defined(SFBENCH_K1OM)

#include <memory>
#include <utility>

#if __cplusplus < 201402L
namespace sfbench_compat {
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace sfbench_compat

#define SFBENCH_MAKE_UNIQUE(Type, ...) sfbench_compat::make_unique<Type>(__VA_ARGS__)
#else
#define SFBENCH_MAKE_UNIQUE(Type, ...) std::make_unique<Type>(__VA_ARGS__)
#endif

#else

#define SFBENCH_MAKE_UNIQUE(Type, ...) std::make_unique<Type>(__VA_ARGS__)

#endif
