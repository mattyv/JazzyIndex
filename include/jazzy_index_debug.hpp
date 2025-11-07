#ifndef JAZZY_INDEX_DEBUG_HPP
#define JAZZY_INDEX_DEBUG_HPP

#include <string>

namespace jazzy {

#ifdef JAZZY_DEBUG_LOGGING

#include <mutex>
#include <cstdio>
#include <cstdarg>

// Thread-safe debug log buffer
class DebugLogger {
private:
    std::string log_buffer_;
    mutable std::mutex mutex_;

    DebugLogger() = default;

public:
    // Get singleton instance
    static DebugLogger& instance() {
        static DebugLogger logger;
        return logger;
    }

    // Prevent copying
    DebugLogger(const DebugLogger&) = delete;
    DebugLogger& operator=(const DebugLogger&) = delete;

    // Append formatted log message
    void log(const char* format, ...) {
        char buffer[1024];
        va_list args;
        va_start(args, format);
        vsnprintf(buffer, sizeof(buffer), format, args);
        va_end(args);

        std::lock_guard<std::mutex> lock(mutex_);
        log_buffer_ += buffer;
        log_buffer_ += '\n';
    }

    // Get current log contents
    std::string get_log() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return log_buffer_;
    }

    // Clear log buffer
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        log_buffer_.clear();
    }
};

// Macro for debug logging
#define DEBUG_LOG(...) jazzy::DebugLogger::instance().log(__VA_ARGS__)

// Helper functions for tests
inline std::string get_debug_log() {
    return DebugLogger::instance().get_log();
}

inline void clear_debug_log() {
    DebugLogger::instance().clear();
}

#else

// When debugging is disabled, these become no-ops
#define DEBUG_LOG(...) ((void)0)

inline std::string get_debug_log() {
    return "";
}

inline void clear_debug_log() {
}

#endif // JAZZY_DEBUG_LOGGING

} // namespace jazzy

#endif // JAZZY_INDEX_DEBUG_HPP
