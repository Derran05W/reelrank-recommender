#pragma once

#include <cstdio>
#include <optional>
#include <string_view>

namespace rr {

// Severity ordering: Trace < Debug < Info < Warn < Error (TDD 25).
enum class LogLevel { Trace, Debug, Info, Warn, Error };

// Parse a level name case-insensitively. Returns std::nullopt for unknown or empty input.
std::optional<LogLevel> parseLogLevel(std::string_view s);

namespace log {

// Current threshold: only messages at this level or higher are emitted.
void setLevel(LogLevel level);
LogLevel level();

// Output destination (default stderr; injectable for tests).
void setSink(std::FILE *sink);

// True iff a message at `level` would currently be emitted.
bool enabled(LogLevel level);

// printf-style emit. Format: "[LEVEL] <message>\n". A no-op if `level` is below the threshold.
void message(LogLevel level, const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

void trace(const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;
void debug(const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;
void info(const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;
void warn(const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;
void error(const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

} // namespace log
} // namespace rr
