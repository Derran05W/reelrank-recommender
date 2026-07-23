// apps/train_models — Phase 22 offline learner CLI (V2 TDD 4.19/4.20, contracts §5, D21).
//
// Reads the frozen §2 training log, assigns a deterministic split, fits the in-house logistic /
// linear models for the requested §4.19 targets, evaluates them against the three baselines on the
// held-out test split, and writes per-target model-<target>.json + calibration-<target>.csv and an
// (appended) training_eval.csv. All heavy lifting lives in rr_learning_v2 (training_data.*, the two
// learners) so this file is only argument parsing + file I/O.
//
// CLI (contracts §5): --log-dir DIR --out-dir DIR --split temporal|user_disjoint --seed N
//                     [--targets csv-list] [--survey]
// Optional SGD overrides (defaults documented in sgd_common.hpp): --epochs N --batch-size N
//                     --lr X --l2 X

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "rr/learning_v2/sgd_common.hpp"
#include "rr/learning_v2/training_data.hpp"

namespace fs = std::filesystem;
using namespace rr::learning_v2;

namespace {

void usage() {
    std::cerr << "usage: train_models --log-dir DIR --out-dir DIR "
                 "--split temporal|user_disjoint --seed N [--targets csv-list] [--survey]\n"
                 "                    [--epochs N] [--batch-size N] [--lr X] [--l2 X]\n";
}

// Split a comma-separated list, trimming empty tokens.
std::vector<std::string> splitCommaList(const std::string &s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) {
            out.push_back(tok);
        }
    }
    return out;
}

} // namespace

int main(int argc, char **argv) {
    std::string logDir;
    std::string outDir;
    std::string splitStr;
    std::string targetsStr;
    bool haveSeed = false;
    uint64_t seed = 0;
    bool survey = false;
    SgdHyperparams hp; // documented defaults

    auto needValue = [&](int &i) -> std::string {
        if (i + 1 >= argc) {
            std::cerr << "train_models: missing value for " << argv[i] << "\n";
            usage();
            std::exit(2);
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--log-dir") {
            logDir = needValue(i);
        } else if (a == "--out-dir") {
            outDir = needValue(i);
        } else if (a == "--split") {
            splitStr = needValue(i);
        } else if (a == "--seed") {
            seed = std::stoull(needValue(i));
            haveSeed = true;
        } else if (a == "--targets") {
            targetsStr = needValue(i);
        } else if (a == "--survey") {
            survey = true;
        } else if (a == "--epochs") {
            hp.epochs = std::stoi(needValue(i));
        } else if (a == "--batch-size") {
            hp.batchSize = std::stoi(needValue(i));
        } else if (a == "--lr") {
            hp.learningRate = std::stod(needValue(i));
        } else if (a == "--l2") {
            hp.l2 = std::stod(needValue(i));
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            std::cerr << "train_models: unknown argument '" << a << "'\n";
            usage();
            return 2;
        }
    }

    if (logDir.empty() || outDir.empty() || splitStr.empty() || !haveSeed) {
        std::cerr << "train_models: --log-dir, --out-dir, --split and --seed are required\n";
        usage();
        return 2;
    }
    hp.seed = seed;

    SplitMode mode;
    try {
        mode = parseSplitMode(splitStr);
    } catch (const std::exception &e) {
        std::cerr << "train_models: " << e.what() << "\n";
        return 2;
    }

    // Resolve the target list.
    std::vector<TargetSpec> targets;
    if (targetsStr.empty()) {
        for (const TargetSpec &t : allTargets()) {
            if (t.requiresSurvey && !survey) {
                continue; // satisfaction only under --survey
            }
            targets.push_back(t);
        }
    } else {
        for (const std::string &name : splitCommaList(targetsStr)) {
            const TargetSpec *t = findTarget(name);
            if (t == nullptr) {
                std::cerr << "train_models: unknown target '" << name << "'\n";
                return 2;
            }
            targets.push_back(*t);
        }
    }

    Dataset ds;
    try {
        ds = loadTrainingLog(logDir, survey);
    } catch (const std::exception &e) {
        std::cerr << "train_models: failed to read training log: " << e.what() << "\n";
        return 1;
    }
    const Split split = assignSplit(ds, mode);

    std::cerr << "train_models: split=" << splitModeName(mode) << " seed=" << seed
              << " rows=" << ds.rows.size() << " (train=" << split.train.size()
              << " test=" << split.test.size() << ") shown=" << ds.shownCandidates
              << " joined=" << ds.joinedWithOutcome << " dropped_no_request=" << ds.droppedNoRequest
              << " survey_rows=" << ds.surveyRows << "\n";

    std::error_code ec;
    fs::create_directories(outDir, ec);
    if (ec) {
        std::cerr << "train_models: cannot create out-dir " << outDir << ": " << ec.message()
                  << "\n";
        return 1;
    }

    const fs::path evalPath = fs::path(outDir) / "training_eval.csv";
    const bool needHeader = !fs::exists(evalPath) || fs::file_size(evalPath) == 0;
    std::ofstream eval(evalPath, std::ios::app);
    if (!eval) {
        std::cerr << "train_models: cannot open " << evalPath << " for writing\n";
        return 1;
    }
    if (needHeader) {
        eval << kTrainingEvalHeader << "\n";
    }

    int trained = 0;
    int skipped = 0;
    for (const TargetSpec &t : targets) {
        if (t.requiresSurvey && !survey) {
            std::cerr << "SKIP target=" << t.name << ": requires --survey\n";
            ++skipped;
            continue;
        }
        const TargetResult r = trainAndEvaluateTarget(ds, split, t, hp, mode);
        if (r.skipped) {
            std::cerr << "SKIP target=" << t.name << ": " << r.skipReason << "\n";
            ++skipped;
            continue;
        }
        // model-<target>.json
        {
            std::ofstream mf(fs::path(outDir) / ("model-" + t.name + ".json"));
            mf << r.modelJson << "\n";
        }
        // calibration-<target>.csv
        {
            std::ofstream cf(fs::path(outDir) / ("calibration-" + t.name + ".csv"));
            cf << kCalibrationHeader << "\n";
            for (const CalibrationBin &b : r.calibration) {
                cf << formatCalibrationBin(b) << "\n";
            }
        }
        for (const EvalRow &row : r.evalRows) {
            eval << formatEvalRow(row) << "\n";
        }
        ++trained;
        // Headline: learned vs the frequency baselines for a quick eyeball.
        const EvalRow &learned = r.evalRows.front();
        std::cerr << "  target=" << t.name << " model=learned auc=" << learned.auc
                  << " log_loss=" << learned.logLoss << " rmse=" << learned.rmse
                  << " base_rate=" << learned.baseRate << "\n";
    }

    std::cerr << "train_models: done — trained=" << trained << " skipped=" << skipped
              << " eval=" << evalPath << "\n";
    return 0;
}
