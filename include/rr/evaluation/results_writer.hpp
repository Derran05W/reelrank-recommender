#pragma once

#include <filesystem>

#include "rr/evaluation/experiment_runner.hpp"
#include "rr/evaluation/run_metadata.hpp"

namespace rr {

// Serializes an ExperimentResult to the §26 output layout in `result.directory`:
//   config.json, summary.json, retrieval_metrics.csv, recommendation_metrics.csv,
//   learning_curve.csv, regret_curve.csv, latency_metrics.csv, metadata.json.
//
// Determinism (D8/TDD 24.6): every file EXCEPT latency_metrics.csv, metadata.json, and the
// `timing` subsection of summary.json is byte-identical across two runs with the same seed. This
// includes retrieval_metrics.csv: live retrieval metrics come from deterministic exact index
// searches. All floating-point output uses fixed precision under the classic locale so formatting
// never depends on the ambient locale.
class ResultsWriter {
  public:
    // Writes every §26 file. `meta` supplies the wall-clock/hardware provenance for metadata.json.
    static void writeAll(const ExperimentResult &result, const RunMetadata &meta);

    // Individual writers (exposed for targeted tests).
    static void writeConfigJson(const ExperimentResult &result);
    static void writeSummaryJson(const ExperimentResult &result);
    static void writeRetrievalMetricsCsv(const ExperimentResult &result);
    static void writeRecommendationMetricsCsv(const ExperimentResult &result);
    static void writeLearningCurveCsv(const ExperimentResult &result);
    static void writeRegretCurveCsv(const ExperimentResult &result);
    static void writeLatencyMetricsCsv(const ExperimentResult &result);
};

} // namespace rr
