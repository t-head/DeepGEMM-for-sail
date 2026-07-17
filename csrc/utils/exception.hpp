#pragma once

#include <exception>
#include <string>
#include <sstream>

namespace deep_gemm {

class DGException final : public std::exception {
    std::string message = {};

public:
    explicit DGException(const char *name, const char* file, const int line, const std::string& error) {
        message = std::string(name) + " error (" + file + ":" + std::to_string(line) + "): " + error;
    }

    const char *what() const noexcept override {
        return message.c_str();
    }
};

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

#ifndef DG_HOST_ASSERT
#define DG_HOST_ASSERT(cond) \
do { \
    if (not (cond)) { \
        throw DGException("Assertion", __FILE__, __LINE__, #cond); \
    } \
} while (0)
#endif

#ifndef DG_HOST_UNREACHABLE
#define DG_HOST_UNREACHABLE(reason) (throw DGException("Assertion", __FILE__, __LINE__, reason))
#endif

#ifndef DG_HGRTC_CHECK
#define DG_HGRTC_CHECK(cmd) \
do { \
    const auto& e = (cmd); \
    if (e != HGRTC_SUCCESS) { \
        throw DGException("HGGCRTC", __FILE__, __LINE__, hgrtcGetErrorString(e)); \
    } \
} while (0)
#endif

#ifndef DG_HGGC_DRIVER_CHECK
#define DG_HGGC_DRIVER_CHECK(cmd) \
do { \
    const auto& e = (cmd); \
    if (e != HGGC_SUCCESS) { \
        std::stringstream ss; \
        const char *name, *info; \
        lazy_hgGetErrorName(e, &name), lazy_hgGetErrorString(e, &info); \
        ss << static_cast<int>(e) << " (" << name << ", " << info << ")"; \
        throw DGException("HGGC driver", __FILE__, __LINE__, ss.str()); \
    } \
} while (0)
#endif

#ifndef DG_HGGC_RUNTIME_CHECK
#define DG_HGGC_RUNTIME_CHECK(cmd) \
do { \
    const auto& e = (cmd); \
    if (e != hggcSuccess) { \
        std::stringstream ss; \
        ss << static_cast<int>(e) << " (" << hggcGetErrorName(e) << ", " << hggcGetErrorString(e) << ")"; \
        throw DGException("HGGC runtime", __FILE__, __LINE__, ss.str()); \
    } \
} while (0)
#endif

} // namespace deep_gemm
