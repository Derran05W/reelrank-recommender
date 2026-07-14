// benchmark_retrieval — standalone HNSW-vs-exact retrieval benchmark (ReelRank Phase 1 task 4,
// extended in Phase 11 package B to complete the TDD 17.3 grid).
//
// Default sweep (no new flags) is the BYTE-COMPATIBLE Phase 1 subset — same seeds, same streams,
// same deterministic outputs:
//   vector counts : {10000, 100000}   (--vector-counts overrides; accepts 1000000)
//   dimensions    : {64}              (--dims LIST sweeps {32,64,128,256})
//   M             : {8, 16, 32}
//   efConstruction: {200}             (--efcs LIST sweeps {50,100,200,400})
//   efSearch      : {16, 32, 64, 128, 256}
//   k             : {10, 50, 200, 500}
//   data          : isotropic random unit vectors (--clustered => generator-backed topic clusters)
//
// The full grid iterates dim x vectorCount x efConstruction x M x efSearch x k. For each
// (dim, vectorCount, efConstruction, M) it builds ONE HNSWVectorIndex (measuring build time /
// insert throughput / RSS delta / peak RSS / graph level distribution / level-0 degree histogram
// once), then sweeps efSearch x k on that same graph via setEfSearch(). Exact ground truth is one
// ExactVectorIndex per (dim, vectorCount); each query's exact top-500 is computed once and sliced
// for Recall@{10,50,200,500}.
//
// --clustered swaps the isotropic random data for topic-clustered reel embeddings from the Phase 2
// generator (rr::generateDataset): the index holds the first `vectorCount` generated reels and the
// query set is held out as generated reels NOT inserted (default, --clustered-query-source=reels)
// or as generated users' hidden preferred-topic mixes (--clustered-query-source=users). This is the
// production-like data for the TDD 27 "Recall@10 > 90%" verdict.
//
// --count-distances builds the HNSW graph with a distance-computation counter (TDD 17.3 "Distance
// computations per query, if feasible") and reports distance_comps_per_query. LATENCY HYGIENE:
// counting adds a relaxed-atomic increment per distance, so a counting row's latency is NOT clean;
// such rows are flagged by distance_comps_per_query >= 0 (rows with -1 were measured counting-off
// and have clean latency). Run the latency sweeps without --count-distances and the distance pass
// with it.
//
// D8 stream-naming rule (keeps existing cells byte-identical): dataset/query streams are
// "dataset-vc<vc>-d<dim>" / "queries-vc<vc>-d<dim>" (reproduces the Phase 1 "...-d64" names exactly
// at dim==64); the HNSW construction seed uses the legacy stream "hnsw-vc<vc>-M<m>" for the Phase 1
// cell (dim==64 AND efConstruction==200) and "hnsw-vc<vc>-M<m>-d<dim>-efc<efc>" for every new cell,
// so no existing cell's seed changes while new cells get independent, reproducible seeds.
//
// Design-decision compliance:
//   D2  — this file lives under apps/ (NOT src/vindex/), so it includes NO vector-db header;
//         everything is reached through rr::HNSWVectorIndex / rr::ExactVectorIndex / rr::*.
//   D7  — performance benchmark, separate apps/ executable, excluded from ctest.
//   D8  — all randomness via rr::Rng / forkRng; no std::*_distribution.
//   D9  — wall-clock only inside rr::Stopwatch for latency; nothing else reads a real clock except
//         the experiment-id timestamp and metadata generation.
//   D12 — custom lightweight harness (warmup + timed samples, p50/p95/p99 from stored samples);
//         writes results/<experiment-id>/{config.json,summary.json,*.csv,metadata.json}.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "rr/core/embedding.hpp"
#include "rr/domain/ids.hpp"
#include "rr/infrastructure/clock.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/process_stats.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/simulation/dataset_generator.hpp"
#include "rr/vindex/exact_vector_index.hpp"
#include "rr/vindex/hnsw_vector_index.hpp"

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>
#elif defined(__linux__)
#include <sys/utsname.h>
#include <unistd.h>
#endif

// Repo/build facts injected by CMake (see apps/CMakeLists.txt).
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

// ---------------------------------------------------------------------------------------------
// Harness helpers (D12). percentile()/currentRssBytes() are the portable prior-art snippets from
// vector-db's bench/mem_stats.hpp, reused verbatim (they have zero vector-db-specific deps, so
// copying them here does NOT violate D2 containment — no vector-db header is included).
// ---------------------------------------------------------------------------------------------

double percentile(std::vector<double> values, double p) {
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const double rank = (p / 100.0) * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(rank));
    const size_t hi = static_cast<size_t>(std::ceil(rank));
    if (lo == hi)
        return values[lo];
    const double frac = rank - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

// Resident set size of the current process, in bytes; 0 when unavailable. Process-wide, so
// per-index deltas are approximate — record before/after pairs and report the delta.
size_t currentRssBytes() {
#if defined(__APPLE__)
    mach_task_basic_info info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info),
                  &count) == KERN_SUCCESS) {
        return static_cast<size_t>(info.resident_size);
    }
    return 0;
#elif defined(__linux__)
    long rss_pages = 0;
    if (FILE *f = std::fopen("/proc/self/statm", "r")) {
        long size_pages = 0;
        if (std::fscanf(f, "%ld %ld", &size_pages, &rss_pages) != 2)
            rss_pages = 0;
        std::fclose(f);
    }
    return rss_pages > 0
               ? static_cast<size_t>(rss_pages) * static_cast<size_t>(sysconf(_SC_PAGESIZE))
               : 0;
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------------------------
// Small utilities.
// ---------------------------------------------------------------------------------------------

// Run a shell command, return trimmed stdout ("" on failure). Used only for git provenance in
// metadata.json — never on any measured hot path.
std::string runCommand(const std::string &cmd) {
    std::string out;
    if (FILE *pipe = ::popen(cmd.c_str(), "r")) {
        char buf[512];
        while (std::fgets(buf, sizeof(buf), pipe) != nullptr)
            out += buf;
        ::pclose(pipe);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) {
        out.pop_back();
    }
    return out;
}

std::string jsonEscape(const std::string &s) {
    std::string r;
    r.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '"':
            r += "\\\"";
            break;
        case '\\':
            r += "\\\\";
            break;
        case '\n':
            r += "\\n";
            break;
        case '\r':
            r += "\\r";
            break;
        case '\t':
            r += "\\t";
            break;
        default:
            r += c;
            break;
        }
    }
    return r;
}

std::string nowStamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d-%H%M%S");
    return oss.str();
}

std::string nowIso() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

#if defined(__APPLE__)
std::string sysctlString(const char *name) {
    size_t len = 0;
    if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0 || len == 0)
        return "";
    std::string buf(len, '\0');
    if (sysctlbyname(name, buf.data(), &len, nullptr, 0) != 0)
        return "";
    if (!buf.empty() && buf.back() == '\0')
        buf.pop_back();
    return buf;
}
uint64_t sysctlU64(const char *name) {
    uint64_t v = 0;
    size_t len = sizeof(v);
    if (sysctlbyname(name, &v, &len, nullptr, 0) != 0)
        return 0;
    return v;
}
#endif

// Generate `count` L2-normalized `dim`-d embeddings with isotropic (gaussian-per-component)
// directions. D8: rr::Rng.gaussian() only — no std::normal_distribution.
std::vector<rr::Embedding> generateEmbeddings(rr::Rng &rng, size_t count, size_t dim) {
    std::vector<rr::Embedding> data;
    data.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        rr::Embedding e(dim);
        for (size_t d = 0; d < dim; ++d)
            e[d] = static_cast<float>(rng.gaussian());
        rr::normalize(e);
        data.push_back(std::move(e));
    }
    return data;
}

// Parse a comma-separated list of non-negative integers ("32,64,128") into `T` values. Rejects
// empties/garbage with a fatal message (a mistyped sweep axis should not silently shrink the grid).
template <class T> std::vector<T> parseList(const std::string &s, const char *flag) {
    std::vector<T> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // trim surrounding whitespace
        size_t b = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (b == std::string::npos)
            continue;
        tok = tok.substr(b, e - b + 1);
        try {
            out.push_back(static_cast<T>(std::stoull(tok)));
        } catch (const std::exception &) {
            std::cerr << "invalid value '" << tok << "' for " << flag << "\n";
            std::exit(2);
        }
    }
    if (out.empty()) {
        std::cerr << "empty list for " << flag << "\n";
        std::exit(2);
    }
    return out;
}

// A (data, queries) pair for one (dim, vectorCount) cell. `data` is inserted into both the exact
// and HNSW indexes; `queries` is the held-out probe set (never inserted).
struct DatasetBundle {
    std::vector<rr::Embedding> data;    // the indexed corpus (size == vectorCount)
    std::vector<rr::Embedding> queries; // held-out probes (size == numQueries)
};

// Isotropic-random data on the Phase-1 stream names (byte-identical at dim==64).
DatasetBundle makeRandomBundle(uint64_t seed, size_t vc, size_t dim, size_t numQueries) {
    rr::Rng dataRng =
        rr::forkRng(seed, "dataset-vc" + std::to_string(vc) + "-d" + std::to_string(dim));
    rr::Rng queryRng =
        rr::forkRng(seed, "queries-vc" + std::to_string(vc) + "-d" + std::to_string(dim));
    DatasetBundle b;
    b.data = generateEmbeddings(dataRng, vc, dim);
    b.queries = generateEmbeddings(queryRng, numQueries, dim);
    return b;
}

// Topic-clustered production-like data (TDD 9.2). Generate vc+numQueries reels via the Phase 2
// generator, index the first vc, hold out the last numQueries reels as probes ("reels" source), or
// use the first numQueries generated users' hidden preferred-topic mixes as probes ("users"
// source). All embeddings are L2-normalized by the generator (D3/D5). Deterministic in (seed, dim,
// vc) — generateDataset forks its own named streams ("topics"/"creators"/"reels"/"users"), disjoint
// from the random-mode streams, so enabling clustered mode never perturbs random-mode output.
DatasetBundle makeClusteredBundle(uint64_t seed, size_t vc, size_t dim, size_t numQueries,
                                  bool usersQuerySource) {
    rr::SimulationConfig scfg;
    scfg.dimensions = static_cast<uint32_t>(dim);
    scfg.topics = 32; // TDD default topic count.
    // Enough creator diversity to matter, bounded so tiny corpora stay cheap.
    scfg.creators = static_cast<uint32_t>(std::clamp<size_t>(vc / 20, 50, 5000));
    scfg.reels = static_cast<uint32_t>(vc + numQueries);
    scfg.users = static_cast<uint32_t>(std::max<size_t>(numQueries, 1));
    scfg.interactionsPerUser = 0; // generateDataset does not simulate; keep it minimal.

    rr::GeneratedDataset ds = rr::generateDataset(scfg, seed);

    DatasetBundle b;
    b.data.reserve(vc);
    for (size_t i = 0; i < vc; ++i)
        b.data.push_back(ds.reels[i].embedding);

    b.queries.reserve(numQueries);
    if (usersQuerySource) {
        for (size_t i = 0; i < numQueries; ++i)
            b.queries.push_back(ds.hiddenStates[i].hiddenPreference);
    } else {
        for (size_t i = 0; i < numQueries; ++i)
            b.queries.push_back(ds.reels[vc + i].embedding);
    }
    return b;
}

// ---------------------------------------------------------------------------------------------
// Result row (one per (vectorCount, M, efSearch, k)).
// ---------------------------------------------------------------------------------------------
struct Row {
    size_t vectorCount;
    size_t dimensions;
    uint32_t m;
    uint32_t efConstruction;
    size_t efSearch;
    size_t k;
    double recallAtK;
    double avgReturned; // mean number of results the ANN actually returned (<= k)
    double p50Ms;
    double p95Ms;
    double p99Ms;
    double meanMs;
    size_t numQueries;
    // per-(dim,vectorCount,efConstruction,M) build facts, repeated across the efSearch x k rows:
    double buildTimeMs;
    double insertThroughput;
    long long memoryDeltaBytes;
    size_t maxGraphLevel;
    size_t numLevels;
    // Phase 11 appended columns:
    std::string dataDistribution; // "random" | "clustered"
    double distanceCompsPerQuery; // mean distance() calls per query; -1.0 when not measured
    double peakRssMb;             // process lifetime peak RSS after this build (rr::peakRssBytes)
};

struct GraphLevelRow {
    size_t vectorCount;
    uint32_t m;
    uint32_t efConstruction;
    size_t maxGraphLevel;
    std::vector<size_t> levelDistribution;
    double buildTimeMs;
    double insertThroughput;
    long long memoryDeltaBytes;
    // Phase 11 appended columns:
    size_t dimensions;
    std::vector<size_t> degreeHistogramLevel0;
    double peakRssMb;
};

} // namespace

int main(int argc, char **argv) {
    // ---- args -----------------------------------------------------------------------------
    uint64_t seed = 42;
    std::string outRoot = "results";
    size_t numQueries = 200;
    bool smoke = false;
    bool clustered = false;
    bool countDistances = false;
    bool clusteredUsersQuerySource = false; // false => held-out reels (default)
    // Empty => apply defaults after parsing (defaults depend on --smoke); non-empty => override.
    std::vector<size_t> dimsArg;
    std::vector<uint32_t> efcsArg;
    std::vector<uint32_t> msArg;
    std::vector<size_t> vectorCountsArg;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char *what) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << what << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--seed") {
            seed = std::stoull(next("--seed"));
        } else if (a == "--out") {
            outRoot = next("--out");
        } else if (a == "--queries") {
            numQueries = std::stoull(next("--queries"));
        } else if (a == "--dims") {
            dimsArg = parseList<size_t>(next("--dims"), "--dims");
        } else if (a == "--efcs") {
            efcsArg = parseList<uint32_t>(next("--efcs"), "--efcs");
        } else if (a == "--ms") {
            msArg = parseList<uint32_t>(next("--ms"), "--ms");
        } else if (a == "--vector-counts") {
            vectorCountsArg = parseList<size_t>(next("--vector-counts"), "--vector-counts");
        } else if (a == "--clustered") {
            clustered = true;
        } else if (a == "--clustered-query-source") {
            const std::string v = next("--clustered-query-source");
            if (v == "users") {
                clusteredUsersQuerySource = true;
            } else if (v == "reels") {
                clusteredUsersQuerySource = false;
            } else {
                std::cerr << "--clustered-query-source must be 'reels' or 'users'\n";
                return 2;
            }
        } else if (a == "--count-distances") {
            countDistances = true;
        } else if (a == "--smoke") {
            smoke = true;
        } else if (a == "--help" || a == "-h") {
            std::cout
                << "usage: benchmark_retrieval [--seed N] [--out DIR] [--queries N] [--smoke]\n"
                   "         [--dims 32,64,128,256] [--efcs 50,100,200,400] [--ms 8,16,32]\n"
                   "         [--vector-counts 10000,100000,1000000]\n"
                   "         [--clustered [--clustered-query-source reels|users]]\n"
                   "         [--count-distances]\n";
            return 0;
        } else {
            std::cerr << "unknown argument: " << a << "\n";
            return 2;
        }
    }

    // ---- grid -----------------------------------------------------------------------------
    // Defaults reproduce the byte-compatible Phase 1 subset (dims {64}, efc {200}); --smoke shrinks
    // the corpus and query count only. User-supplied lists override the corresponding default.
    if (smoke)
        numQueries = 30;
    const std::vector<size_t> dims = dimsArg.empty() ? std::vector<size_t>{64} : dimsArg;
    const std::vector<uint32_t> efcs = efcsArg.empty() ? std::vector<uint32_t>{200} : efcsArg;
    const std::vector<size_t> vectorCounts = !vectorCountsArg.empty() ? vectorCountsArg
                                             : smoke                  ? std::vector<size_t>{1000}
                                                     : std::vector<size_t>{10000, 100000};
    const std::vector<uint32_t> ms = msArg.empty() ? std::vector<uint32_t>{8, 16, 32} : msArg;
    const std::vector<size_t> efSearches = {16, 32, 64, 128, 256};
    const std::vector<size_t> ks = {10, 50, 200, 500};
    const size_t warmupQueries = std::min<size_t>(10, numQueries);
    const std::string dataDistribution = clustered ? "clustered" : "random";

    const std::string expId = "hnsw_retrieval-seed" + std::to_string(seed) + "-" + nowStamp();
    const std::filesystem::path outDir = std::filesystem::path(outRoot) / expId;
    std::filesystem::create_directories(outDir);

    std::cerr << "benchmark_retrieval: experiment-id=" << expId << "\n"
              << "  out=" << outDir.string() << "\n"
              << "  dims=" << dims.size() << " efcs=" << efcs.size()
              << " vector_counts=" << vectorCounts.size() << " data=" << dataDistribution
              << " count_distances=" << (countDistances ? "on" : "off") << " queries=" << numQueries
              << (smoke ? "  [SMOKE]\n" : "\n");

    rr::Stopwatch wallClock; // total sweep wall time.

    std::vector<Row> rows;
    std::vector<GraphLevelRow> levelRows;
    const size_t maxK = *std::max_element(ks.begin(), ks.end());

    // ---- sweep: dim x vectorCount x efConstruction x M x efSearch x k ----------------------
    for (size_t dim : dims) {
        for (size_t vc : vectorCounts) {
            std::cerr << "[d=" << dim << " vc=" << vc << "] generating " << dataDistribution
                      << " dataset + queries ..." << std::endl;
            const DatasetBundle bundle = clustered ? makeClusteredBundle(seed, vc, dim, numQueries,
                                                                         clusteredUsersQuerySource)
                                                   : makeRandomBundle(seed, vc, dim, numQueries);
            const std::vector<rr::Embedding> &data = bundle.data;
            const std::vector<rr::Embedding> &queries = bundle.queries;

            // Exact ground truth: one index per (dim, vc); precompute each query's exact top-500.
            std::cerr << "[d=" << dim << " vc=" << vc
                      << "] building ExactVectorIndex ground truth ..." << std::endl;
            rr::ExactVectorIndex exact(dim);
            for (size_t i = 0; i < data.size(); ++i) {
                exact.insert(rr::ReelId{static_cast<uint32_t>(i)}, data[i]);
            }
            std::vector<std::vector<uint32_t>> exactTop(queries.size());
            for (size_t q = 0; q < queries.size(); ++q) {
                const auto res = exact.search(queries[q], maxK);
                exactTop[q].reserve(res.size());
                for (const auto &r : res)
                    exactTop[q].push_back(r.reelId.value);
            }

            for (uint32_t efc : efcs) {
                for (uint32_t m : ms) {
                    // Build ONE HNSW graph for this (dim, vc, efc, m). Freed before the next
                    // cell so the RSS delta captures just this graph.
                    rr::HNSWConfig cfg;
                    cfg.m = m;
                    cfg.efConstruction = efc;
                    cfg.efSearch = 64; // overridden per-efSearch below via setEfSearch().
                    // D8 naming: legacy stream for the Phase 1 cell (dim==64 AND efc==200), a
                    // dim/efc-qualified stream for every new cell (see file header).
                    std::string hnswStream =
                        "hnsw-vc" + std::to_string(vc) + "-M" + std::to_string(m);
                    if (!(dim == 64 && efc == 200)) {
                        hnswStream += "-d" + std::to_string(dim) + "-efc" + std::to_string(efc);
                    }
                    const uint64_t hnswSeed = rr::splitmix64(seed ^ rr::fnv1a64(hnswStream));

                    std::cerr << "[d=" << dim << " vc=" << vc << " efc=" << efc << " M=" << m
                              << "] building HNSW graph ..." << std::endl;
                    auto hnsw =
                        std::make_unique<rr::HNSWVectorIndex>(dim, cfg, hnswSeed, countDistances);

                    const size_t rssBefore = currentRssBytes();
                    rr::Stopwatch buildTimer;
                    for (size_t i = 0; i < data.size(); ++i) {
                        hnsw->insert(rr::ReelId{static_cast<uint32_t>(i)}, data[i]);
                    }
                    const double buildTimeMs = buildTimer.elapsedMs();
                    const size_t rssAfter = currentRssBytes();
                    const long long memoryDeltaBytes =
                        static_cast<long long>(rssAfter) - static_cast<long long>(rssBefore);
                    // Process lifetime peak RSS after this build (D12/TDD 18.7 memory usage; the
                    // meaningful figure at 1M where the delta method is noisy).
                    const double peakRssMb =
                        static_cast<double>(rr::peakRssBytes()) / (1024.0 * 1024.0);
                    const double insertThroughput =
                        buildTimeMs > 0.0
                            ? (static_cast<double>(data.size()) / (buildTimeMs / 1000.0))
                            : 0.0;

                    const rr::HnswGraphStats gs = hnsw->graphStats();
                    const std::vector<size_t> &levelDist = gs.levelDistribution;
                    const size_t numLevels = levelDist.size();
                    const size_t maxGraphLevel = numLevels == 0 ? 0 : numLevels - 1;

                    GraphLevelRow lr;
                    lr.vectorCount = vc;
                    lr.m = m;
                    lr.efConstruction = efc;
                    lr.maxGraphLevel = maxGraphLevel;
                    lr.levelDistribution = levelDist;
                    lr.buildTimeMs = buildTimeMs;
                    lr.insertThroughput = insertThroughput;
                    lr.memoryDeltaBytes = memoryDeltaBytes;
                    lr.dimensions = dim;
                    lr.degreeHistogramLevel0 = gs.degreeHistogramLevel0;
                    lr.peakRssMb = peakRssMb;
                    levelRows.push_back(std::move(lr));

                    std::cerr << "[d=" << dim << " vc=" << vc << " efc=" << efc << " M=" << m
                              << "] build " << std::fixed << std::setprecision(1) << buildTimeMs
                              << " ms, " << static_cast<long long>(insertThroughput)
                              << " inserts/s, maxLevel=" << maxGraphLevel << ", RSS delta "
                              << (memoryDeltaBytes / (1024 * 1024)) << " MB, peak "
                              << static_cast<long long>(peakRssMb) << " MB" << std::endl;

                    for (size_t ef : efSearches) {
                        hnsw->setEfSearch(ef);
                        for (size_t k : ks) {
                            // warmup (untimed). search() returns a heap-allocated vector
                            // (observable side effect), so it is not optimized away when discarded.
                            for (size_t w = 0; w < warmupQueries; ++w) {
                                auto discard = hnsw->search(queries[w % queries.size()], k);
                                (void)discard;
                            }
                            // Isolate the timed phase for distance counting: zero the counter
                            // after warmup so distance_comps_per_query reflects only timed queries.
                            hnsw->resetDistanceCounter();
                            // timed: one search per query, one latency sample each; recall on same.
                            std::vector<double> latencies;
                            latencies.reserve(queries.size());
                            double recallSum = 0.0;
                            double returnedSum = 0.0;
                            for (size_t q = 0; q < queries.size(); ++q) {
                                rr::Stopwatch qt;
                                const auto approx = hnsw->search(queries[q], k);
                                latencies.push_back(qt.elapsedMs());

                                // exact top-k ids (slice of precomputed top-maxK)
                                const size_t kk = std::min(k, exactTop[q].size());
                                std::unordered_set<uint32_t> exactSet(
                                    exactTop[q].begin(),
                                    exactTop[q].begin() + static_cast<std::ptrdiff_t>(kk));
                                size_t overlap = 0;
                                for (const auto &r : approx) {
                                    if (exactSet.count(r.reelId.value))
                                        ++overlap;
                                }
                                recallSum +=
                                    (k > 0) ? (static_cast<double>(overlap) /
                                               static_cast<double>(std::min(k, exactSet.size())))
                                            : 0.0;
                                returnedSum += static_cast<double>(approx.size());
                            }

                            Row row;
                            row.vectorCount = vc;
                            row.dimensions = dim;
                            row.m = m;
                            row.efConstruction = efc;
                            row.efSearch = ef;
                            row.k = k;
                            row.recallAtK = recallSum / static_cast<double>(queries.size());
                            row.avgReturned = returnedSum / static_cast<double>(queries.size());
                            row.p50Ms = percentile(latencies, 50.0);
                            row.p95Ms = percentile(latencies, 95.0);
                            row.p99Ms = percentile(latencies, 99.0);
                            double sum = 0.0;
                            for (double l : latencies)
                                sum += l;
                            row.meanMs = sum / static_cast<double>(latencies.size());
                            row.numQueries = queries.size();
                            row.buildTimeMs = buildTimeMs;
                            row.insertThroughput = insertThroughput;
                            row.memoryDeltaBytes = memoryDeltaBytes;
                            row.maxGraphLevel = maxGraphLevel;
                            row.numLevels = numLevels;
                            row.dataDistribution = dataDistribution;
                            // Mean distance() calls per query (build excluded via the reset above);
                            // -1 flags "not measured" AND "latency is clean" (see file header).
                            const uint64_t distComps = hnsw->distanceComputations();
                            row.distanceCompsPerQuery =
                                countDistances ? static_cast<double>(distComps) /
                                                     static_cast<double>(queries.size())
                                               : -1.0;
                            row.peakRssMb = peakRssMb;
                            rows.push_back(row);
                        }
                    }
                } // M (hnsw freed here)
            } // efConstruction
        } // vectorCount
    } // dim

    const double totalWallMs = wallClock.elapsedMs();

    // ---- write retrieval_metrics.csv ------------------------------------------------------
    {
        std::ofstream csv(outDir / "retrieval_metrics.csv");
        // Phase-1 prefix columns kept in order (downstream tooling depends on it); Phase 11 appends
        // data_distribution, distance_comps_per_query (-1 = not measured / clean latency),
        // peak_rss_mb.
        csv << "vector_count,dimensions,m,ef_construction,ef_search,k,recall_at_k,avg_returned,"
               "query_p50_ms,query_p95_ms,query_p99_ms,query_mean_ms,num_queries,build_time_ms,"
               "insert_throughput_per_sec,memory_delta_bytes,memory_delta_mb,max_graph_level,"
               "num_levels,data_distribution,distance_comps_per_query,peak_rss_mb\n";
        csv << std::fixed;
        for (const auto &r : rows) {
            csv << r.vectorCount << ',' << r.dimensions << ',' << r.m << ',' << r.efConstruction
                << ',' << r.efSearch << ',' << r.k << ',' << std::setprecision(6) << r.recallAtK
                << ',' << std::setprecision(3) << r.avgReturned << ',' << std::setprecision(6)
                << r.p50Ms << ',' << r.p95Ms << ',' << r.p99Ms << ',' << r.meanMs << ','
                << r.numQueries << ',' << std::setprecision(3) << r.buildTimeMs << ','
                << std::setprecision(2) << r.insertThroughput << ',' << r.memoryDeltaBytes << ','
                << std::setprecision(3)
                << (static_cast<double>(r.memoryDeltaBytes) / (1024.0 * 1024.0)) << ','
                << r.maxGraphLevel << ',' << r.numLevels << ',' << r.dataDistribution << ','
                << std::setprecision(2) << r.distanceCompsPerQuery << ',' << r.peakRssMb << '\n';
        }
    }

    // ---- write graph_levels.csv -----------------------------------------------------------
    {
        std::ofstream csv(outDir / "graph_levels.csv");
        // Phase-1 prefix columns kept in order; Phase 11 appends dimensions,
        // degree_histogram_level0 (";"-joined; index d = nodes with exactly d level-0 nbrs),
        // peak_rss_mb.
        csv << "vector_count,m,ef_construction,max_graph_level,build_time_ms,"
               "insert_throughput_per_sec,memory_delta_mb,level_distribution,dimensions,"
               "degree_histogram_level0,peak_rss_mb\n";
        csv << std::fixed;
        for (const auto &g : levelRows) {
            csv << g.vectorCount << ',' << g.m << ',' << g.efConstruction << ',' << g.maxGraphLevel
                << ',' << std::setprecision(3) << g.buildTimeMs << ',' << std::setprecision(2)
                << g.insertThroughput << ',' << std::setprecision(3)
                << (static_cast<double>(g.memoryDeltaBytes) / (1024.0 * 1024.0)) << ',';
            // level distribution as a semicolon-joined list "l0;l1;l2;..."
            for (size_t i = 0; i < g.levelDistribution.size(); ++i) {
                if (i)
                    csv << ';';
                csv << g.levelDistribution[i];
            }
            csv << ',' << g.dimensions << ',';
            // level-0 degree histogram as a semicolon-joined list "d0;d1;d2;..."
            for (size_t i = 0; i < g.degreeHistogramLevel0.size(); ++i) {
                if (i)
                    csv << ';';
                csv << g.degreeHistogramLevel0[i];
            }
            csv << ',' << std::setprecision(2) << g.peakRssMb << '\n';
        }
    }

    // ---- write config.json ----------------------------------------------------------------
    {
        std::ofstream j(outDir / "config.json");
        auto arr = [](const auto &v) {
            std::ostringstream o;
            o << '[';
            for (size_t i = 0; i < v.size(); ++i) {
                if (i)
                    o << ',';
                o << v[i];
            }
            o << ']';
            return o.str();
        };
        j << "{\n";
        j << "  \"experiment_id\": \"" << expId << "\",\n";
        j << "  \"name\": \"hnsw_retrieval\",\n";
        j << "  \"seed\": " << seed << ",\n";
        j << "  \"dimensions\": " << arr(dims) << ",\n";
        j << "  \"vector_counts\": " << arr(vectorCounts) << ",\n";
        j << "  \"m_values\": " << arr(ms) << ",\n";
        j << "  \"ef_construction_values\": " << arr(efcs) << ",\n";
        j << "  \"ef_search_values\": " << arr(efSearches) << ",\n";
        j << "  \"k_values\": " << arr(ks) << ",\n";
        j << "  \"num_queries\": " << numQueries << ",\n";
        j << "  \"warmup_queries\": " << warmupQueries << ",\n";
        j << "  \"data_distribution\": \"" << dataDistribution << "\",\n";
        j << "  \"clustered_query_source\": \""
          << (clustered ? (clusteredUsersQuerySource ? "users" : "reels") : "n/a") << "\",\n";
        j << "  \"count_distances\": " << (countDistances ? "true" : "false") << ",\n";
        j << "  \"distance_metric\": \"euclidean_on_unit_vectors (cosine-equivalent, D3)\",\n";
        j << "  \"embedding_generation\": \""
          << (clustered ? "topic-clustered reels via rr::generateDataset (Phase 2, TDD 9.2)"
                        : "gaussian-per-component, L2-normalized (D8 rr::Rng)")
          << "\"\n";
        j << "}\n";
    }

    // ---- write summary.json (headline aggregates) -----------------------------------------
    // Generalized for the Phase 11 grid: one "per_build" entry per (dim, vc, efc, M) build cell
    // (unambiguous), plus a "per_size" recall/latency range per vector_count over the whole sweep.
    {
        std::ofstream j(outDir / "summary.json");
        j << std::fixed;
        j << "{\n";
        j << "  \"experiment_id\": \"" << expId << "\",\n";
        j << "  \"data_distribution\": \"" << dataDistribution << "\",\n";
        j << "  \"count_distances\": " << (countDistances ? "true" : "false") << ",\n";
        j << "  \"total_sweep_wall_ms\": " << std::setprecision(1) << totalWallMs << ",\n";
        j << "  \"total_rows\": " << rows.size() << ",\n";
        j << "  \"per_size\": [\n";
        for (size_t vi = 0; vi < vectorCounts.size(); ++vi) {
            const size_t vc = vectorCounts[vi];
            double r10min = 1e9, r10max = -1e9;
            double p50min = 1e9, p50max = -1e9;
            double thrMin = 1e18, thrMax = -1e18;
            for (const auto &r : rows) {
                if (r.vectorCount != vc)
                    continue;
                if (r.k == 10u) {
                    r10min = std::min(r10min, r.recallAtK);
                    r10max = std::max(r10max, r.recallAtK);
                }
                p50min = std::min(p50min, r.p50Ms);
                p50max = std::max(p50max, r.p50Ms);
                thrMin = std::min(thrMin, r.insertThroughput);
                thrMax = std::max(thrMax, r.insertThroughput);
            }
            j << "    {\n";
            j << "      \"vector_count\": " << vc << ",\n";
            j << "      \"recall_at_10_min\": " << std::setprecision(4) << r10min << ",\n";
            j << "      \"recall_at_10_max\": " << std::setprecision(4) << r10max << ",\n";
            j << "      \"query_p50_ms_min\": " << std::setprecision(6) << p50min << ",\n";
            j << "      \"query_p50_ms_max\": " << std::setprecision(6) << p50max << ",\n";
            j << "      \"insert_throughput_min\": " << std::setprecision(1) << thrMin << ",\n";
            j << "      \"insert_throughput_max\": " << std::setprecision(1) << thrMax << "\n";
            j << "    }" << (vi + 1 < vectorCounts.size() ? "," : "") << "\n";
        }
        j << "  ],\n";
        j << "  \"per_build\": [\n";
        for (size_t gi = 0; gi < levelRows.size(); ++gi) {
            const GraphLevelRow &g = levelRows[gi];
            // recall@10 at efSearch extremes for this exact build cell.
            double r10ef16 = -1.0, r10ef256 = -1.0;
            for (const auto &r : rows) {
                if (r.dimensions == g.dimensions && r.vectorCount == g.vectorCount &&
                    r.efConstruction == g.efConstruction && r.m == g.m && r.k == 10u) {
                    if (r.efSearch == 16u)
                        r10ef16 = r.recallAtK;
                    if (r.efSearch == 256u)
                        r10ef256 = r.recallAtK;
                }
            }
            j << "    {\n";
            j << "      \"dimensions\": " << g.dimensions << ",\n";
            j << "      \"vector_count\": " << g.vectorCount << ",\n";
            j << "      \"ef_construction\": " << g.efConstruction << ",\n";
            j << "      \"m\": " << g.m << ",\n";
            j << "      \"build_time_ms\": " << std::setprecision(1) << g.buildTimeMs << ",\n";
            j << "      \"insert_throughput_per_sec\": " << std::setprecision(1)
              << g.insertThroughput << ",\n";
            j << "      \"max_graph_level\": " << g.maxGraphLevel << ",\n";
            j << "      \"memory_delta_mb\": " << std::setprecision(1)
              << (static_cast<double>(g.memoryDeltaBytes) / (1024.0 * 1024.0)) << ",\n";
            j << "      \"peak_rss_mb\": " << std::setprecision(1) << g.peakRssMb << ",\n";
            j << "      \"recall_at_10_ef16\": " << std::setprecision(4) << r10ef16 << ",\n";
            j << "      \"recall_at_10_ef256\": " << std::setprecision(4) << r10ef256 << "\n";
            j << "    }" << (gi + 1 < levelRows.size() ? "," : "") << "\n";
        }
        j << "  ]\n";
        j << "}\n";
    }

    // ---- write metadata.json --------------------------------------------------------------
    {
        const std::string repoDir = RR_REPO_DIR;
        const std::string vdbDir = RR_VDB_DIR;
        const std::string rrSha = runCommand("git -C '" + repoDir + "' rev-parse HEAD 2>/dev/null");
        const std::string rrDirtyRaw =
            runCommand("git -C '" + repoDir + "' status --porcelain 2>/dev/null");
        const bool rrDirty = !rrDirtyRaw.empty();
        const std::string vdbSha = runCommand("git -C '" + vdbDir + "' rev-parse HEAD 2>/dev/null");

#if defined(__APPLE__)
        const std::string cpu = sysctlString("machdep.cpu.brand_string");
        const uint64_t ramBytes = sysctlU64("hw.memsize");
        const uint64_t logicalCores = sysctlU64("hw.ncpu");
        const uint64_t physicalCores = sysctlU64("hw.physicalcpu");
#else
        const std::string cpu = "unknown";
        const uint64_t ramBytes = 0;
        const uint64_t logicalCores = 0;
        const uint64_t physicalCores = 0;
#endif
        std::string osSys, osRel, osVer, osMach;
#if defined(__APPLE__) || defined(__linux__)
        struct utsname uts{};
        if (uname(&uts) == 0) {
            osSys = uts.sysname;
            osRel = uts.release;
            osVer = uts.version;
            osMach = uts.machine;
        }
#endif

        std::ofstream j(outDir / "metadata.json");
        j << "{\n";
        j << "  \"experiment_id\": \"" << expId << "\",\n";
        j << "  \"generated_at\": \"" << nowIso() << "\",\n";
        j << "  \"tool\": \"benchmark_retrieval (ReelRank Phase 1, Phase 11 grid completion)\",\n";
        j << "  \"git\": {\n";
        j << "    \"reel_rank_sha\": \"" << jsonEscape(rrSha) << "\",\n";
        j << "    \"reel_rank_dir\": \"" << jsonEscape(repoDir) << "\",\n";
        j << "    \"reel_rank_dirty\": " << (rrDirty ? "true" : "false") << ",\n";
        j << "    \"vector_db_sha\": \"" << jsonEscape(vdbSha) << "\",\n";
        j << "    \"vector_db_dir\": \"" << jsonEscape(vdbDir) << "\",\n";
        j << "    \"note\": \"reel-rank SHA is the Phase 0 commit; Phase 1 work (this benchmark, "
             "the vindex adapters, the differential suite) is uncommitted, so the tree is dirty at "
             "benchmark time.\"\n";
        j << "  },\n";
        j << "  \"build\": {\n";
        j << "    \"type\": \"" << jsonEscape(RR_BUILD_TYPE) << "\",\n";
        j << "    \"compiler\": \"" << jsonEscape(RR_COMPILER) << "\",\n";
        j << "    \"cxx_standard\": 20,\n";
        j << "    \"release_flags\": \"-O3 -DNDEBUG -Wall -Wextra -Wpedantic -Werror\"\n";
        j << "  },\n";
        j << "  \"hardware\": {\n";
        j << "    \"cpu_model\": \"" << jsonEscape(cpu) << "\",\n";
        j << "    \"logical_cores\": " << logicalCores << ",\n";
        j << "    \"physical_cores\": " << physicalCores << ",\n";
        j << "    \"ram_bytes\": " << ramBytes << ",\n";
        j << "    \"ram_gib\": " << std::fixed << std::setprecision(1)
          << (static_cast<double>(ramBytes) / (1024.0 * 1024.0 * 1024.0)) << "\n";
        j << "  },\n";
        j << "  \"os\": {\n";
        j << "    \"sysname\": \"" << jsonEscape(osSys) << "\",\n";
        j << "    \"release\": \"" << jsonEscape(osRel) << "\",\n";
        j << "    \"version\": \"" << jsonEscape(osVer) << "\",\n";
        j << "    \"machine\": \"" << jsonEscape(osMach) << "\"\n";
        j << "  },\n";
        j << "  \"execution\": {\n";
        j << "    \"threads_used\": 1,\n";
        j << "    \"note\": \"single-threaded benchmark (D13); logical_cores reported for "
             "provenance only\"\n";
        j << "  },\n";
        auto jarr = [&](const auto &v) {
            j << '[';
            for (size_t i = 0; i < v.size(); ++i) {
                if (i)
                    j << ',';
                j << v[i];
            }
            j << ']';
        };
        j << "  \"sweep\": {\n";
        j << "    \"seed\": " << seed << ",\n";
        j << "    \"dimensions\": ";
        jarr(dims);
        j << ",\n";
        j << "    \"vector_counts\": ";
        jarr(vectorCounts);
        j << ",\n";
        j << "    \"m_values\": ";
        jarr(ms);
        j << ",\n";
        j << "    \"ef_construction_values\": ";
        jarr(efcs);
        j << ",\n";
        j << "    \"ef_search_values\": [16,32,64,128,256],\n";
        j << "    \"k_values\": [10,50,200,500],\n";
        j << "    \"num_queries\": " << numQueries << ",\n";
        j << "    \"data_distribution\": \"" << dataDistribution << "\",\n";
        j << "    \"count_distances\": " << (countDistances ? "true" : "false") << ",\n";
        j << "    \"total_sweep_wall_ms\": " << std::setprecision(1) << totalWallMs << ",\n";
        j << "    \"subset_note\": \"Phase 11 package B completes the TDD 17.3 grid: dimensions, "
             "efConstruction, the 1,000,000 corpus, topic-clustered data, and level-0 degree "
             "histogram + distance-computations-per-query are all reachable via CLI flags. The "
             "DEFAULT invocation (no flags) reproduces the byte-compatible Phase 1 subset (dims "
             "{64}, efConstruction {200}, vector counts {10000,100000}, isotropic random data). "
             "The axes swept in THIS run are the arrays above.\"\n";
        j << "  }\n";
        j << "}\n";
    }

    std::cerr << "benchmark_retrieval: DONE in " << std::fixed << std::setprecision(1)
              << (totalWallMs / 1000.0) << " s; wrote " << rows.size() << " rows to "
              << outDir.string() << std::endl;
    // One-line machine-parseable pointer for the caller.
    std::cout << "RESULT_DIR=" << outDir.string() << "\n";
    return 0;
}
