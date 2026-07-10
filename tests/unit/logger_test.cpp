#include "rr/infrastructure/logger.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <functional>
#include <optional>
#include <string>

namespace {

// Redirect the logger to an in-memory temp file, run body, return everything written.
std::string capture(rr::LogLevel level, const std::function<void()> &body) {
    FILE *tmp = std::tmpfile();
    rr::log::setSink(tmp);
    rr::log::setLevel(level);
    body();
    std::fflush(tmp);
    std::rewind(tmp);
    std::string out;
    char buf[512];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), tmp)) > 0) {
        out.append(buf, n);
    }
    std::fclose(tmp);
    rr::log::setSink(stderr);
    return out;
}

} // namespace

TEST(LoggerTest, WarnLevelSuppressesInfo) {
    std::string out = capture(rr::LogLevel::Warn, [] {
        rr::log::info("hidden");
        rr::log::error("shown");
    });
    EXPECT_EQ(out.find("hidden"), std::string::npos);
    EXPECT_NE(out.find("shown"), std::string::npos);
    EXPECT_NE(out.find("[ERROR]"), std::string::npos);
}

TEST(LoggerTest, TraceLevelWritesEverything) {
    std::string out = capture(rr::LogLevel::Trace, [] {
        rr::log::trace("t");
        rr::log::debug("d");
        rr::log::info("i");
        rr::log::warn("w");
        rr::log::error("e");
    });
    EXPECT_NE(out.find("[TRACE]"), std::string::npos);
    EXPECT_NE(out.find("[DEBUG]"), std::string::npos);
    EXPECT_NE(out.find("[INFO]"), std::string::npos);
    EXPECT_NE(out.find("[WARN]"), std::string::npos);
    EXPECT_NE(out.find("[ERROR]"), std::string::npos);
}

TEST(LoggerTest, FormatsArguments) {
    std::string out =
        capture(rr::LogLevel::Info, [] { rr::log::info("value=%d name=%s", 7, "x"); });
    EXPECT_NE(out.find("value=7 name=x"), std::string::npos);
}

TEST(LoggerTest, EnabledReflectsLevel) {
    rr::log::setLevel(rr::LogLevel::Warn);
    EXPECT_FALSE(rr::log::enabled(rr::LogLevel::Info));
    EXPECT_TRUE(rr::log::enabled(rr::LogLevel::Warn));
    EXPECT_TRUE(rr::log::enabled(rr::LogLevel::Error));
    rr::log::setLevel(rr::LogLevel::Info); // restore default
}

TEST(LoggerTest, ParseLogLevel) {
    EXPECT_EQ(rr::parseLogLevel("trace"), rr::LogLevel::Trace);
    EXPECT_EQ(rr::parseLogLevel("DEBUG"), rr::LogLevel::Debug);
    EXPECT_EQ(rr::parseLogLevel("Info"), rr::LogLevel::Info);
    EXPECT_EQ(rr::parseLogLevel("WARN"), rr::LogLevel::Warn);
    EXPECT_EQ(rr::parseLogLevel("error"), rr::LogLevel::Error);
    EXPECT_FALSE(rr::parseLogLevel("nonsense").has_value());
    EXPECT_FALSE(rr::parseLogLevel("").has_value());
}
