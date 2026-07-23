#pragma once

// ================================================================================================
// Phase 22 package B — SHARED TEST FIXTURE (header-only, inline). Package A owns the real log
// EMISSION; package B works from the FROZEN schema by fabricating schema-conformant CSVs with a
// KNOWN planted structure. Two generators live here:
//
//   1. writePlantedLog(dir, params): writes requests/candidates/outcomes(+survey).csv whose HEADERS
//      come straight from training_log_schema.hpp (so they cannot drift from the frozen names) and
//      whose outcomes are drawn from a KNOWN logistic/linear function of three feature columns plus
//      noise. Used by the offline "learned beats frequency baselines" statistical test and the
//      reader tests. `followed` is planted RARE (< kMinPositivesToTrain) to exercise the SKIP path.
//
//   2. makeSeparableBinary / makeLinearTarget: tiny in-memory (X, y) sets with a hand-checkable
//      signal for the learner convergence unit tests.
//
// Determinism: every draw is a rr::Rng (D8). No std::*_distribution.
// ================================================================================================

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "rr/infrastructure/random.hpp"
#include "rr/learning_v2/sgd_common.hpp"
#include "rr/learning_v2/training_log_schema.hpp"

namespace rr::learning_v2::test {

namespace fs = std::filesystem;

// ---- in-memory generators for the convergence unit tests ---------------------------------------

struct XY {
    std::vector<std::vector<double>> x;
    std::vector<double> y;
};

// Binary data separable on feature 0 (a clean logistic threshold at 0.5); all other features are
// zero so the standardizer drops them. AUC on held-out data should be ~1.0 after training.
inline XY makeSeparableBinary(int n, uint64_t seed) {
    rr::Rng rng(seed);
    XY d;
    for (int i = 0; i < n; ++i) {
        std::vector<double> row(kNumFeatures, 0.0);
        const double f0 = rng.uniform01();
        row[0] = f0;
        d.x.push_back(row);
        d.y.push_back(f0 > 0.5 ? 1.0 : 0.0);
    }
    return d;
}

// Continuous target y = 0.3 + 1.5*f0 - 0.8*f1 (+ tiny noise). A linear model recovers it to low
// RMSE and calibration slope ~1.
inline XY makeLinearTarget(int n, uint64_t seed) {
    rr::Rng rng(seed);
    XY d;
    for (int i = 0; i < n; ++i) {
        std::vector<double> row(kNumFeatures, 0.0);
        const double f0 = rng.uniform01();
        const double f1 = rng.uniform01();
        row[0] = f0;
        row[1] = f1;
        d.x.push_back(row);
        d.y.push_back(0.3 + 1.5 * f0 - 0.8 * f1 + 0.02 * rng.gaussian());
    }
    return d;
}

// ---- planted-log CSV generator -----------------------------------------------------------------

struct PlantedLogParams {
    uint64_t seed = 7;
    int requests = 1000;
    int shownPerRequest = 4;
    int poolOnlyPerRequest = 2; // shown=0, position=-1 rows with NO outcome (reader must drop them)
    int numUsers = 300;
    bool survey = false;
    double surveyRate = 0.5; // dense on purpose so the satisfaction target has samples to learn
};

namespace detail {

inline std::string d2s(double x) {
    std::ostringstream o;
    o << std::setprecision(9) << x;
    return o.str();
}

template <std::size_t N>
inline void writeHeader(std::ostream &os, const std::array<std::string_view, N> &header) {
    for (std::size_t i = 0; i < N; ++i) {
        if (i) {
            os << ',';
        }
        os << header[i];
    }
    os << "\n";
}

template <std::size_t N>
inline void writeRow(std::ostream &os, const std::array<std::string_view, N> &header,
                     const std::unordered_map<std::string, std::string> &vals) {
    for (std::size_t i = 0; i < N; ++i) {
        if (i) {
            os << ',';
        }
        auto it = vals.find(std::string(header[i]));
        os << (it != vals.end() ? it->second : "0");
    }
    os << "\n";
}

} // namespace detail

inline fs::path makeTempDir(const std::string &tag) {
    static std::atomic<uint64_t> counter{0};
    const fs::path dir =
        fs::temp_directory_path() / ("rr_p22_" + tag + "_" + std::to_string(counter.fetch_add(1)) +
                                     "_" + std::to_string(reinterpret_cast<uintptr_t>(&counter)));
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

// Writes a schema-conformant training log under `dir`. The planted outcome models (documented at
// the call site of the statistical test): completed/liked/shared/session_exit/not_interested are
// logistic in centered {similarity(f0), quality(f2), creator_affinity(f6)}; watch_ratio is linear
// in them; `followed` is deliberately rare; retrieval_sources is assigned INDEPENDENTLY of the
// outcome (so the per-source baseline sits at chance and the learned-vs-baseline margin is robust).
inline void writePlantedLog(const fs::path &dir, const PlantedLogParams &p) {
    fs::create_directories(dir);
    std::ofstream reqOut(dir / "requests.csv");
    std::ofstream candOut(dir / "candidates.csv");
    std::ofstream outOut(dir / "outcomes.csv");
    std::ofstream surveyOut;
    if (p.survey) {
        surveyOut.open(dir / "survey.csv");
    }

    detail::writeHeader(reqOut, kRequestsColumns);
    // candidates header = prefix columns then feature columns (schema order).
    for (std::size_t i = 0; i < kCandidatesPrefixColumns.size(); ++i) {
        candOut << kCandidatesPrefixColumns[i] << ',';
    }
    for (std::size_t i = 0; i < kFeatureColumns.size(); ++i) {
        if (i) {
            candOut << ',';
        }
        candOut << kFeatureColumns[i];
    }
    candOut << "\n";
    detail::writeHeader(outOut, kOutcomesColumns);
    if (p.survey) {
        detail::writeHeader(surveyOut, kSurveyColumns);
    }

    static const std::array<std::string_view, 4> kSources = {"vector_hnsw", "trending",
                                                             "creator_affinity", "exploration"};
    rr::Rng rng(p.seed);
    uint64_t reelCounter = 1;

    for (int r = 0; r < p.requests; ++r) {
        const uint64_t requestId = static_cast<uint64_t>(r) + 1;
        const uint64_t userId = static_cast<uint64_t>(r % p.numUsers) + 1;
        const uint64_t timestamp = 1'000'000ULL + static_cast<uint64_t>(r); // monotone for temporal

        {
            std::unordered_map<std::string, std::string> v;
            v["request_id"] = std::to_string(requestId);
            v["user_id"] = std::to_string(userId);
            v["session_id"] = std::to_string(userId * 10 + 1);
            v["timestamp"] = std::to_string(timestamp);
            v["feed_size"] = std::to_string(p.shownPerRequest);
            v["effective_epsilon"] = "0.1";
            v["pool_size"] = std::to_string(p.shownPerRequest + p.poolOnlyPerRequest);
            v["shown_count"] = std::to_string(p.shownPerRequest);
            v["pool_logged"] = "1";
            detail::writeRow(reqOut, kRequestsColumns, v);
        }

        const int total = p.shownPerRequest + p.poolOnlyPerRequest;
        for (int c = 0; c < total; ++c) {
            const bool shown = c < p.shownPerRequest;
            const uint64_t reelId = reelCounter++;
            const std::string src = std::string(kSources[reelId % kSources.size()]);

            // Feature vector: signal-bearing columns are uniform01; the V2 tail stays zero (mimics
            // a gate-off log so the constant-feature standardization path is exercised).
            std::array<double, kNumFeatures> f{};
            for (std::size_t k = 0; k <= 10; ++k) {
                f[k] = rng.uniform01();
            }
            f[7] = (src == "exploration") ? 1.0 : 0.0; // exploration flag mirrors the source

            const double f0 = f[0] - 0.5; // similarity, centered
            const double f2 = f[2] - 0.5; // quality
            const double f6 = f[6] - 0.5; // creator_affinity

            auto bern = [&](double logit) { return rng.bernoulli(sigmoid(logit)) ? 1 : 0; };
            const int completed = bern(0.0 + 3.0 * f0 + 2.5 * f2);
            const int liked = bern(-1.0 + 2.5 * f2 + 2.0 * f6);
            const int shared = bern(-2.0 + 2.2 * f0);
            const int notInterested = bern(-1.5 - 2.0 * f2);
            const int sessionExit = bern(-1.0 + 2.0 * f0);
            const int followed = bern(-6.5 + 1.5 * f6); // deliberately rare -> SKIP path
            double watchRatio = 0.5 + 0.4 * f0 + 0.3 * f2 + 0.05 * rng.gaussian();
            watchRatio = std::clamp(watchRatio, 0.0, 1.0);
            // served_score is a NOISY proxy of the completed logit (a plausibly-decent ranker); it
            // is reported but never part of the hard learned-vs-baseline assertion (P21 lesson).
            const double servedScore = 3.0 * f0 + 2.5 * f2 + 0.8 * rng.gaussian();

            {
                std::unordered_map<std::string, std::string> v;
                v["request_id"] = std::to_string(requestId);
                v["reel_id"] = std::to_string(reelId);
                v["pool_rank"] = std::to_string(c);
                v["shown"] = shown ? "1" : "0";
                v["position"] = shown ? std::to_string(c) : "-1";
                v["served_score"] = detail::d2s(servedScore);
                v["exploration_flag"] = (src == "exploration") ? "1" : "0";
                v["retrieval_sources"] = src;
                v["retrieval_similarity"] = detail::d2s(f[0]);
                for (std::size_t k = 0; k < kFeatureColumns.size(); ++k) {
                    v[std::string(kFeatureColumns[k])] = detail::d2s(f[k]);
                }
                // Emit prefix then features in schema order.
                for (std::size_t i = 0; i < kCandidatesPrefixColumns.size(); ++i) {
                    candOut << v[std::string(kCandidatesPrefixColumns[i])] << ',';
                }
                for (std::size_t i = 0; i < kFeatureColumns.size(); ++i) {
                    if (i) {
                        candOut << ',';
                    }
                    candOut << v[std::string(kFeatureColumns[i])];
                }
                candOut << "\n";
            }

            if (!shown) {
                continue; // pool-only row: no outcome (reader must drop it from training)
            }

            {
                std::unordered_map<std::string, std::string> v;
                v["request_id"] = std::to_string(requestId);
                v["reel_id"] = std::to_string(reelId);
                v["position"] = std::to_string(c);
                v["watch_seconds"] = detail::d2s(watchRatio * 30.0);
                v["watch_ratio"] = detail::d2s(watchRatio);
                v["completed"] = std::to_string(completed);
                v["liked"] = std::to_string(liked);
                v["shared"] = std::to_string(shared);
                v["followed"] = std::to_string(followed);
                v["not_interested"] = std::to_string(notInterested);
                v["commented"] = "0";
                v["saved"] = "0";
                v["profile_visited"] = "0";
                v["observed_exit_after_impression"] = std::to_string(sessionExit);
                detail::writeRow(outOut, kOutcomesColumns, v);
            }

            if (p.survey && rng.bernoulli(p.surveyRate)) {
                // likert tracks watch_ratio (satisfaction proxy) with quantization noise.
                double likert = 1.0 + 4.0 * watchRatio + 0.4 * rng.gaussian();
                likert = std::clamp(likert, 1.0, 5.0);
                const long q = std::lround(likert);
                std::unordered_map<std::string, std::string> v;
                v["user_id"] = std::to_string(userId);
                v["reel_id"] = std::to_string(reelId);
                v["request_id"] = std::to_string(requestId);
                v["timestamp"] = std::to_string(timestamp);
                v["likert"] = std::to_string(q);
                detail::writeRow(surveyOut, kSurveyColumns, v);
            }
        }
    }
}

} // namespace rr::learning_v2::test
