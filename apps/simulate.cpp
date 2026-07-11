// simulate — end-to-end ReelRank evaluation harness CLI (Phase 4, task 6).
//
// Loads a fully-resolved ExperimentConfig, runs the ExperimentRunner (dataset generation ->
// cold-start -> interleaved request/feed/interaction loop -> metrics + oracle regret), and writes
// the §26 output layout under <out>/<experiment-id>/. Prints only a few headline lines.
//
//   simulate --config <path> --algorithm <name> --seed <N> --out <dir> [--smoke]
//
// --algorithm / --seed override the config when given. --smoke shrinks the dataset to a tiny CI
// config that finishes in a few seconds (D14). Design-decision compliance: all randomness flows
// through rr::Rng named streams (D8); wall clock appears only in latency/timing provenance (D9).

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include "rr/evaluation/experiment_runner.hpp"
#include "rr/evaluation/run_metadata.hpp"
#include "rr/infrastructure/config.hpp"

// Repo/build facts injected by CMake (see apps/CMakeLists.txt) for metadata.json provenance.
#ifndef RR_REPO_DIR
#define RR_REPO_DIR "unknown"
#endif
#ifndef RR_VDB_DIR
#define RR_VDB_DIR "unknown"
#endif
#ifndef RR_BUILD_TYPE
#define RR_BUILD_TYPE "unknown"
#endif
#ifndef RR_COMPILER
#define RR_COMPILER "unknown"
#endif

namespace {

void printUsage(std::ostream &os) {
    os << "usage: simulate [--config <path>] [--algorithm <name>] [--seed <N>] [--out <dir>] "
          "[--smoke]\n"
          "  --config     experiment config JSON (default: configs/small.json; optional with "
          "--smoke)\n"
          "  --algorithm  random | popularity | exact_vector | hnsw (overrides config)\n"
          "  --seed       master random seed (overrides config.simulation.seed)\n"
          "  --out        results root directory (default: results)\n"
          "  --smoke      tiny CI config: 50 users, 500 reels, 25 creators, 8 topics, dim 32, "
          "10 interactions/user\n";
}

// Apply the tiny smoke-test dataset overrides (D14). Must finish in a few seconds.
void applySmoke(rr::ExperimentConfig &config) {
    config.simulation.users = 50;
    config.simulation.reels = 500;
    config.simulation.creators = 25;
    config.simulation.topics = 8;
    config.simulation.dimensions = 32;
    config.simulation.interactionsPerUser = 10;
    config.evaluation.oracleSampleRate = 0.5;
    // Sample every request for live retrieval metrics at smoke scale so the exact-vs-exact recall
    // self-check (recall 1.0 / distance error 0.0) is always exercised for vector algorithms.
    config.evaluation.retrievalSampleRate = 1.0;
}

} // namespace

int main(int argc, char **argv) {
    std::string configPath;
    std::string algorithm;
    std::string outDir = "results";
    uint64_t seed = 0;
    bool haveSeed = false;
    bool smoke = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char *what) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "error: missing value for " << what << "\n";
                printUsage(std::cerr);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--config") {
            configPath = next("--config");
        } else if (a == "--algorithm") {
            algorithm = next("--algorithm");
        } else if (a == "--seed") {
            seed = std::stoull(next("--seed"));
            haveSeed = true;
        } else if (a == "--out") {
            outDir = next("--out");
        } else if (a == "--smoke") {
            smoke = true;
        } else if (a == "--help" || a == "-h") {
            printUsage(std::cout);
            return 0;
        } else {
            std::cerr << "error: unknown argument: " << a << "\n";
            printUsage(std::cerr);
            return 2;
        }
    }

    try {
        rr::ExperimentConfig config;
        // A config is required unless --smoke supplies a self-contained default dataset.
        if (!configPath.empty()) {
            config = rr::loadExperimentConfig(configPath);
        } else if (!smoke) {
            configPath = "configs/small.json";
            config = rr::loadExperimentConfig(configPath);
        }

        if (smoke) {
            applySmoke(config);
        }
        if (!algorithm.empty()) {
            config.algorithm = rr::algorithmFromString(algorithm);
        }
        if (haveSeed) {
            config.simulation.seed = seed;
        }

        rr::BuildProvenance provenance;
        provenance.repoDir = RR_REPO_DIR;
        provenance.vdbDir = RR_VDB_DIR;
        provenance.buildType = RR_BUILD_TYPE;
        provenance.compiler = RR_COMPILER;

        rr::ExperimentRunner runner(config, outDir, provenance);
        const rr::ExperimentResult result = runner.run();

        const rr::MetricsSummary &m = result.overall;
        std::cout << "simulate: " << rr::toString(config.algorithm) << (smoke ? " [SMOKE]" : "")
                  << "\n";
        std::cout << "  out                  " << result.directory.string() << "\n";
        std::cout << "  seed                 " << result.seed << "\n";
        std::cout << "  users/reels          " << result.userCount << " / " << result.reelCount
                  << "\n";
        std::cout << "  requests/impressions " << result.requestCount << " / "
                  << result.impressionCount << "\n";
        std::cout << "  completion_rate      " << m.completionRate << "\n";
        std::cout << "  reward_per_impression " << m.rewardPerImpression << "\n";
        std::cout << "  mean_true_affinity   " << m.meanTrueAffinity << "\n";
        std::cout << "  mean_regret          " << result.meanRegret << " (over "
                  << result.sampledRequestCount << " sampled requests)\n";
        // Live retrieval headline (TDD 18.1), only when retrieval samples were taken (vector-based
        // algorithm with a positive sample rate).
        if (result.retrievalSampleCount > 0) {
            std::cout << "  recall@10 / @50      " << result.retrievalRecallAt10 << " / "
                      << result.retrievalRecallAt50 << "\n";
            std::cout << "  mean_distance_error  " << result.retrievalDistanceError << "\n";
            std::cout << "  retrieval p50/p95 ms " << result.retrievalLatency.p50Ms << " / "
                      << result.retrievalLatency.p95Ms << " (over " << result.retrievalSampleCount
                      << " sampled requests)\n";
        }
        std::cout << "  wall_seconds         " << result.totalWallSeconds << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
