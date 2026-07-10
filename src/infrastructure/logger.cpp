#include "rr/infrastructure/logger.hpp"

#include <cctype>
#include <cstdarg>
#include <cstdlib>
#include <string>

namespace rr {

std::optional<LogLevel> parseLogLevel(std::string_view s) {
    std::string lower;
    lower.reserve(s.size());
    for (char c : s) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower == "trace") {
        return LogLevel::Trace;
    }
    if (lower == "debug") {
        return LogLevel::Debug;
    }
    if (lower == "info") {
        return LogLevel::Info;
    }
    if (lower == "warn") {
        return LogLevel::Warn;
    }
    if (lower == "error") {
        return LogLevel::Error;
    }
    return std::nullopt;
}

namespace log {
namespace {

const char *levelTag(LogLevel level) {
    switch (level) {
    case LogLevel::Trace:
        return "TRACE";
    case LogLevel::Debug:
        return "DEBUG";
    case LogLevel::Info:
        return "INFO";
    case LogLevel::Warn:
        return "WARN";
    case LogLevel::Error:
        return "ERROR";
    }
    return "INFO";
}

struct State {
    LogLevel level = LogLevel::Info;
    std::FILE *sink = stderr;

    State() {
        // One-time initialization from RR_LOG_LEVEL (default INFO; junk keeps default + warns).
        if (const char *env = std::getenv("RR_LOG_LEVEL")) {
            if (auto parsed = parseLogLevel(env)) {
                level = *parsed;
            } else {
                std::fprintf(sink, "[WARN] ignoring unknown RR_LOG_LEVEL=%s\n", env);
            }
        }
    }
};

State &state() {
    static State s;
    return s;
}

void vemit(LogLevel level, const char *fmt, std::va_list args) {
    if (!enabled(level)) {
        return;
    }
    std::FILE *sink = state().sink;
    std::fprintf(sink, "[%s] ", levelTag(level));
    std::vfprintf(sink, fmt, args);
    std::fputc('\n', sink);
}

} // namespace

void setLevel(LogLevel level) { state().level = level; }

LogLevel level() { return state().level; }

void setSink(std::FILE *sink) { state().sink = sink; }

bool enabled(LogLevel level) { return static_cast<int>(level) >= static_cast<int>(state().level); }

void message(LogLevel level, const char *fmt, ...) {
    std::va_list args;
    va_start(args, fmt);
    vemit(level, fmt, args);
    va_end(args);
}

void trace(const char *fmt, ...) {
    std::va_list args;
    va_start(args, fmt);
    vemit(LogLevel::Trace, fmt, args);
    va_end(args);
}

void debug(const char *fmt, ...) {
    std::va_list args;
    va_start(args, fmt);
    vemit(LogLevel::Debug, fmt, args);
    va_end(args);
}

void info(const char *fmt, ...) {
    std::va_list args;
    va_start(args, fmt);
    vemit(LogLevel::Info, fmt, args);
    va_end(args);
}

void warn(const char *fmt, ...) {
    std::va_list args;
    va_start(args, fmt);
    vemit(LogLevel::Warn, fmt, args);
    va_end(args);
}

void error(const char *fmt, ...) {
    std::va_list args;
    va_start(args, fmt);
    vemit(LogLevel::Error, fmt, args);
    va_end(args);
}

} // namespace log
} // namespace rr
