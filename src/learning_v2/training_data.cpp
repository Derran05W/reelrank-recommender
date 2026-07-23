#include "rr/learning_v2/training_data.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

#include "rr/core/pinned_hash.hpp"
#include "rr/learning_v2/linear_regression.hpp"
#include "rr/learning_v2/logistic_regression.hpp"
#include "rr/learning_v2/training_log_schema.hpp"

namespace fs = std::filesystem;

namespace rr::learning_v2 {
namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

// --- CSV plumbing (tolerances documented in training_data.hpp) ----------------------------------

// Simple comma split with NO quoting (schema fields never embed commas). A field that runs off the
// end of a ragged row is read as "".
void splitCsv(const std::string &line, std::vector<std::string> &out) {
    out.clear();
    std::size_t start = 0;
    for (std::size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ',') {
            out.emplace_back(line.substr(start, i - start));
            start = i + 1;
        }
    }
}

bool readLine(std::istream &in, std::string &line) {
    if (!std::getline(in, line)) {
        return false;
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    return true;
}

std::unordered_map<std::string, std::size_t> headerIndex(const std::string &headerLine) {
    std::vector<std::string> cols;
    splitCsv(headerLine, cols);
    std::unordered_map<std::string, std::size_t> idx;
    for (std::size_t i = 0; i < cols.size(); ++i) {
        idx.emplace(cols[i], i);
    }
    return idx;
}

std::size_t need(const std::unordered_map<std::string, std::size_t> &h, std::string_view col,
                 const fs::path &file) {
    auto it = h.find(std::string(col));
    if (it == h.end()) {
        throw std::runtime_error("training_log: required column '" + std::string(col) +
                                 "' missing from " + file.string());
    }
    return it->second;
}

const std::string &cell(const std::vector<std::string> &f, std::size_t i) {
    static const std::string empty;
    return i < f.size() ? f[i] : empty;
}

// Empty / unparseable / non-finite numeric cell -> 0.0 (documented NaN handling).
double toDouble(const std::string &s) {
    if (s.empty()) {
        return 0.0;
    }
    try {
        const double v = std::stod(s);
        return std::isfinite(v) ? v : 0.0;
    } catch (...) {
        return 0.0;
    }
}

uint64_t toU64(const std::string &s) {
    if (s.empty()) {
        return 0;
    }
    try {
        return std::stoull(s);
    } catch (...) {
        return 0;
    }
}

double toBinary(const std::string &s) { return toDouble(s) > 0.5 ? 1.0 : 0.0; }

// Files for a logical table: "<base>.csv" plus rotation parts "<base>-partNNNN.csv",
// filename-sorted for deterministic read order.
std::vector<fs::path> globParts(const fs::path &dir, const std::string &base) {
    std::vector<fs::path> out;
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        return out;
    }
    const std::string exact = base + ".csv";
    const std::string partPrefix = base + "-part";
    for (const auto &entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        const bool isExact = (name == exact);
        const bool isPart = name.starts_with(partPrefix) && name.ends_with(".csv");
        if (isExact || isPart) {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end(),
              [](const fs::path &a, const fs::path &b) { return a.filename() < b.filename(); });
    return out;
}

// Full-precision, CSV-safe number formatting; NaN -> "nan" (pandas reads it as NaN).
std::string fmtNum(double x) {
    if (std::isnan(x)) {
        return "nan";
    }
    std::ostringstream o;
    o << std::setprecision(10) << x;
    return o.str();
}

double meanOf(const std::vector<double> &v) {
    if (v.empty()) {
        return 0.0;
    }
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

using Key = std::pair<uint64_t, uint64_t>;
struct KeyHash {
    std::size_t operator()(const Key &k) const {
        uint64_t h = k.first * 0x9E3779B97F4A7C15ULL;
        h ^= k.second + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
        return static_cast<std::size_t>(h);
    }
};

// Fill a binary EvalRow's metric fields. `isProbability` is false for the served_score baseline
// (a ranking score, not a calibrated probability) -> AUC is still meaningful, log-loss is not.
void fillBinaryMetrics(EvalRow &r, const std::vector<double> &pred,
                       const std::vector<double> &label, bool isProbability) {
    r.auc = rankAuc(pred, label);
    r.logLoss = isProbability ? logLoss(pred, label) : kNaN;
    r.rmse = kNaN;
    r.calSlope = kNaN;
    r.calIntercept = kNaN;
}

// Fill a linear EvalRow's metric fields. `scaleMatched` is false for the served_score baseline
// (different scale from the target) -> RMSE/calibration inapplicable.
void fillLinearMetrics(EvalRow &r, const std::vector<double> &pred,
                       const std::vector<double> &actual, bool scaleMatched) {
    r.auc = kNaN;
    r.logLoss = kNaN;
    if (scaleMatched) {
        r.rmse = rmse(pred, actual);
        const CalibrationFit c = calibrationFit(pred, actual);
        r.calSlope = c.slope;
        r.calIntercept = c.intercept;
    } else {
        r.rmse = kNaN;
        r.calSlope = kNaN;
        r.calIntercept = kNaN;
    }
}

EvalRow makeRow(const std::string &target, const std::string &model, const std::string &split,
                int nTrain, int nTest, const std::vector<double> &pred,
                const std::vector<double> &label, TargetKind kind, bool matched, double baseRate) {
    EvalRow r;
    r.target = target;
    r.model = model;
    r.split = split;
    r.nTrain = nTrain;
    r.nTest = nTest;
    r.baseRate = baseRate;
    if (kind == TargetKind::Binary) {
        fillBinaryMetrics(r, pred, label, matched);
    } else {
        fillLinearMetrics(r, pred, label, matched);
    }
    return r;
}

} // namespace

// --- feature helpers ----------------------------------------------------------------------------

const std::vector<std::string> &featureColumnNames() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> v;
        v.reserve(kFeatureColumns.size());
        for (std::string_view c : kFeatureColumns) {
            v.emplace_back(c);
        }
        return v;
    }();
    return names;
}

std::vector<double> featuresOf(const Example &e) {
    return std::vector<double>(e.features.begin(), e.features.end());
}

// --- targets ------------------------------------------------------------------------------------

const std::vector<TargetSpec> &allTargets() {
    static const std::vector<TargetSpec> kTargets = {
        {"completed", TargetKind::Binary, false},      {"liked", TargetKind::Binary, false},
        {"shared", TargetKind::Binary, false},         {"followed", TargetKind::Binary, false},
        {"not_interested", TargetKind::Binary, false}, {"session_exit", TargetKind::Binary, false},
        {"watch_ratio", TargetKind::Linear, false},    {"satisfaction", TargetKind::Linear, true},
    };
    return kTargets;
}

const TargetSpec *findTarget(std::string_view name) {
    for (const TargetSpec &t : allTargets()) {
        if (t.name == name) {
            return &t;
        }
    }
    return nullptr;
}

std::optional<double> extractTarget(const Example &e, const TargetSpec &t) {
    const std::string &n = t.name;
    if (n == "completed") {
        return e.completed;
    }
    if (n == "liked") {
        return e.liked;
    }
    if (n == "shared") {
        return e.shared;
    }
    if (n == "followed") {
        return e.followed;
    }
    if (n == "not_interested") {
        return e.notInterested;
    }
    if (n == "session_exit") {
        return e.sessionExit;
    }
    if (n == "watch_ratio") {
        return e.watchRatio;
    }
    if (n == "satisfaction") {
        if (!e.hasSurvey) {
            return std::nullopt;
        }
        return e.likert;
    }
    return std::nullopt;
}

// --- splits -------------------------------------------------------------------------------------

SplitMode parseSplitMode(std::string_view s) {
    if (s == "temporal") {
        return SplitMode::Temporal;
    }
    if (s == "user_disjoint") {
        return SplitMode::UserDisjoint;
    }
    throw std::invalid_argument("unknown --split '" + std::string(s) +
                                "' (expected temporal|user_disjoint)");
}

std::string_view splitModeName(SplitMode m) {
    return m == SplitMode::Temporal ? "temporal" : "user_disjoint";
}

Split assignSplit(const Dataset &ds, SplitMode mode) {
    Split s;
    if (mode == SplitMode::Temporal) {
        // Rank distinct requests by (timestamp, request_id); the earliest kTrainFraction are train.
        std::unordered_map<uint64_t, uint64_t> reqTs;
        for (const Example &e : ds.rows) {
            reqTs[e.requestId] = e.timestamp;
        }
        std::vector<std::pair<uint64_t, uint64_t>> reqs; // (timestamp, request_id)
        reqs.reserve(reqTs.size());
        for (const auto &[req, ts] : reqTs) {
            reqs.emplace_back(ts, req);
        }
        std::sort(reqs.begin(), reqs.end());
        const std::size_t nTrain =
            static_cast<std::size_t>(std::floor(kTrainFraction * static_cast<double>(reqs.size())));
        std::unordered_set<uint64_t> trainReq;
        trainReq.reserve(nTrain * 2 + 1);
        for (std::size_t i = 0; i < nTrain; ++i) {
            trainReq.insert(reqs[i].second);
        }
        for (std::size_t i = 0; i < ds.rows.size(); ++i) {
            (trainReq.count(ds.rows[i].requestId) ? s.train : s.test).push_back(i);
        }
    } else {
        for (std::size_t i = 0; i < ds.rows.size(); ++i) {
            const bool train = pinnedHash01(ds.rows[i].userId ^ kSplitSalt) < kTrainFraction;
            (train ? s.train : s.test).push_back(i);
        }
    }
    return s;
}

std::string majoritySourceKey(const std::string &retrievalSources) {
    // Tokenize on the delimiters A might use inside the field (never a comma — that is the CSV
    // separator). Count tokens; return the majority, ties broken lexicographically.
    auto isDelim = [](char c) {
        return c == '|' || c == '+' || c == ';' || c == '/' || c == ' ' || c == '\t';
    };
    std::unordered_map<std::string, int> counts;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= retrievalSources.size(); ++i) {
        if (i == retrievalSources.size() || isDelim(retrievalSources[i])) {
            if (i > start) {
                ++counts[retrievalSources.substr(start, i - start)];
            }
            start = i + 1;
        }
    }
    if (counts.empty()) {
        return "unknown";
    }
    std::string best;
    int bestCount = -1;
    for (const auto &[tok, c] : counts) {
        if (c > bestCount || (c == bestCount && tok < best)) {
            best = tok;
            bestCount = c;
        }
    }
    return best;
}

// --- metrics ------------------------------------------------------------------------------------

double rankAuc(const std::vector<double> &pred, const std::vector<double> &label) {
    const std::size_t n = pred.size();
    if (n == 0) {
        return kNaN;
    }
    double nPos = 0.0;
    for (double y : label) {
        nPos += (y > 0.5) ? 1.0 : 0.0;
    }
    const double nNeg = static_cast<double>(n) - nPos;
    if (nPos == 0.0 || nNeg == 0.0) {
        return kNaN;
    }
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return pred[a] < pred[b]; });
    std::vector<double> rank(n);
    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i;
        while (j < n && pred[order[j]] == pred[order[i]]) {
            ++j;
        }
        // Ranks are 1-based i+1 .. j; tied group gets their average.
        const double avg = (static_cast<double>(i + 1) + static_cast<double>(j)) / 2.0;
        for (std::size_t k = i; k < j; ++k) {
            rank[order[k]] = avg;
        }
        i = j;
    }
    double sumRankPos = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        if (label[k] > 0.5) {
            sumRankPos += rank[k];
        }
    }
    return (sumRankPos - nPos * (nPos + 1.0) / 2.0) / (nPos * nNeg);
}

double logLoss(const std::vector<double> &pred, const std::vector<double> &label) {
    if (pred.empty()) {
        return kNaN;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < pred.size(); ++i) {
        const double p = std::clamp(pred[i], 1e-6, 1.0 - 1e-6);
        const double y = label[i];
        sum += -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
    }
    return sum / static_cast<double>(pred.size());
}

double rmse(const std::vector<double> &pred, const std::vector<double> &actual) {
    if (pred.empty()) {
        return kNaN;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < pred.size(); ++i) {
        const double d = pred[i] - actual[i];
        sum += d * d;
    }
    return std::sqrt(sum / static_cast<double>(pred.size()));
}

CalibrationFit calibrationFit(const std::vector<double> &pred, const std::vector<double> &actual) {
    if (pred.size() < 2) {
        return {kNaN, kNaN};
    }
    const double mp = meanOf(pred);
    const double ma = meanOf(actual);
    double cov = 0.0;
    double var = 0.0;
    for (std::size_t i = 0; i < pred.size(); ++i) {
        const double dp = pred[i] - mp;
        cov += dp * (actual[i] - ma);
        var += dp * dp;
    }
    if (var < 1e-12) {
        return {kNaN, kNaN}; // constant predictor: slope undefined
    }
    const double slope = cov / var;
    return {slope, ma - slope * mp};
}

std::vector<CalibrationBin> equalCountBins(const std::vector<double> &pred,
                                           const std::vector<double> &actual, int nbins) {
    std::vector<CalibrationBin> bins;
    const std::size_t n = pred.size();
    if (n == 0 || nbins <= 0) {
        return bins;
    }
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return pred[a] < pred[b]; });
    const std::size_t nb = std::min(static_cast<std::size_t>(nbins), n);
    for (std::size_t b = 0; b < nb; ++b) {
        const std::size_t start = b * n / nb;
        const std::size_t end = (b + 1) * n / nb;
        if (end <= start) {
            continue;
        }
        double sp = 0.0;
        double sa = 0.0;
        for (std::size_t k = start; k < end; ++k) {
            sp += pred[order[k]];
            sa += actual[order[k]];
        }
        const double cnt = static_cast<double>(end - start);
        bins.push_back({static_cast<int>(b), sp / cnt, sa / cnt, static_cast<int>(end - start)});
    }
    return bins;
}

std::string formatEvalRow(const EvalRow &r) {
    std::ostringstream o;
    o << r.target << ',' << r.model << ',' << r.split << ',' << r.nTrain << ',' << r.nTest << ','
      << fmtNum(r.auc) << ',' << fmtNum(r.logLoss) << ',' << fmtNum(r.rmse) << ','
      << fmtNum(r.calSlope) << ',' << fmtNum(r.calIntercept) << ',' << fmtNum(r.baseRate);
    return o.str();
}

std::string formatCalibrationBin(const CalibrationBin &b) {
    std::ostringstream o;
    o << b.bin << ',' << fmtNum(b.meanPred) << ',' << fmtNum(b.meanActual) << ',' << b.count;
    return o.str();
}

// --- loader -------------------------------------------------------------------------------------

Dataset loadTrainingLog(const fs::path &dir, bool withSurvey) {
    struct CandRow {
        std::array<double, kNumFeatures> features{};
        double servedScore = 0.0;
        std::string sources;
    };
    struct OutRow {
        double watchRatio = 0.0;
        double completed = 0.0, liked = 0.0, shared = 0.0, followed = 0.0, notInterested = 0.0,
               sessionExit = 0.0;
    };
    struct ReqRow {
        uint64_t userId = 0;
        uint64_t timestamp = 0;
    };

    const std::vector<fs::path> candFiles = globParts(dir, "candidates");
    const std::vector<fs::path> outFiles = globParts(dir, "outcomes");
    const std::vector<fs::path> reqFiles = globParts(dir, "requests");
    if (candFiles.empty()) {
        throw std::runtime_error("training_log: no candidates*.csv under " + dir.string());
    }
    if (outFiles.empty()) {
        throw std::runtime_error("training_log: no outcomes*.csv under " + dir.string());
    }
    if (reqFiles.empty()) {
        throw std::runtime_error("training_log: no requests*.csv under " + dir.string());
    }

    // requests: request_id -> (user_id, timestamp)
    std::unordered_map<uint64_t, ReqRow> reqs;
    for (const auto &file : reqFiles) {
        std::ifstream in(file);
        std::string line;
        if (!readLine(in, line)) {
            continue;
        }
        auto h = headerIndex(line);
        const std::size_t iReq = need(h, "request_id", file);
        const std::size_t iUser = need(h, "user_id", file);
        const std::size_t iTs = need(h, "timestamp", file);
        std::vector<std::string> f;
        while (readLine(in, line)) {
            if (line.empty()) {
                continue;
            }
            splitCsv(line, f);
            reqs[toU64(cell(f, iReq))] = {toU64(cell(f, iUser)), toU64(cell(f, iTs))};
        }
    }

    // outcomes: (request_id, reel_id) -> labels
    std::unordered_map<Key, OutRow, KeyHash> out;
    for (const auto &file : outFiles) {
        std::ifstream in(file);
        std::string line;
        if (!readLine(in, line)) {
            continue;
        }
        auto h = headerIndex(line);
        const std::size_t iReq = need(h, "request_id", file);
        const std::size_t iReel = need(h, "reel_id", file);
        const std::size_t iWatch = need(h, "watch_ratio", file);
        const std::size_t iComp = need(h, "completed", file);
        const std::size_t iLike = need(h, "liked", file);
        const std::size_t iShare = need(h, "shared", file);
        const std::size_t iFollow = need(h, "followed", file);
        const std::size_t iNI = need(h, "not_interested", file);
        const std::size_t iExit = need(h, "observed_exit_after_impression", file);
        std::vector<std::string> f;
        while (readLine(in, line)) {
            if (line.empty()) {
                continue;
            }
            splitCsv(line, f);
            OutRow o;
            o.watchRatio = toDouble(cell(f, iWatch));
            o.completed = toBinary(cell(f, iComp));
            o.liked = toBinary(cell(f, iLike));
            o.shared = toBinary(cell(f, iShare));
            o.followed = toBinary(cell(f, iFollow));
            o.notInterested = toBinary(cell(f, iNI));
            o.sessionExit = toBinary(cell(f, iExit));
            out[{toU64(cell(f, iReq)), toU64(cell(f, iReel))}] = o;
        }
    }

    // survey (optional): (request_id, reel_id) -> likert
    std::unordered_map<Key, double, KeyHash> survey;
    if (withSurvey) {
        for (const auto &file : globParts(dir, "survey")) {
            std::ifstream in(file);
            std::string line;
            if (!readLine(in, line)) {
                continue;
            }
            auto h = headerIndex(line);
            const std::size_t iReq = need(h, "request_id", file);
            const std::size_t iReel = need(h, "reel_id", file);
            const std::size_t iLikert = need(h, "likert", file);
            std::vector<std::string> f;
            while (readLine(in, line)) {
                if (line.empty()) {
                    continue;
                }
                splitCsv(line, f);
                survey[{toU64(cell(f, iReq)), toU64(cell(f, iReel))}] = toDouble(cell(f, iLikert));
            }
        }
    }

    // candidates (SHOWN rows only) joined against outcomes + requests.
    Dataset ds;
    for (const auto &file : candFiles) {
        std::ifstream in(file);
        std::string line;
        if (!readLine(in, line)) {
            continue;
        }
        auto h = headerIndex(line);
        const std::size_t iReq = need(h, "request_id", file);
        const std::size_t iReel = need(h, "reel_id", file);
        const std::size_t iShown = need(h, "shown", file);
        const std::size_t iScore = need(h, "served_score", file);
        const std::size_t iSrc = need(h, "retrieval_sources", file);
        std::array<std::size_t, kNumFeatures> iFeat{};
        for (std::size_t k = 0; k < kNumFeatures; ++k) {
            iFeat[k] = need(h, kFeatureColumns[k], file);
        }
        std::vector<std::string> f;
        while (readLine(in, line)) {
            if (line.empty()) {
                continue;
            }
            splitCsv(line, f);
            if (toDouble(cell(f, iShown)) <= 0.5) {
                continue; // shown rows only for training
            }
            ++ds.shownCandidates;
            const Key key{toU64(cell(f, iReq)), toU64(cell(f, iReel))};
            auto oit = out.find(key);
            if (oit == out.end()) {
                continue; // no outcome label to train against
            }
            ++ds.joinedWithOutcome;
            auto rit = reqs.find(key.first);
            if (rit == reqs.end()) {
                ++ds.droppedNoRequest; // cannot assign a split without request metadata
                continue;
            }
            Example e;
            e.requestId = key.first;
            e.reelId = key.second;
            e.userId = rit->second.userId;
            e.timestamp = rit->second.timestamp;
            for (std::size_t k = 0; k < kNumFeatures; ++k) {
                e.features[k] = toDouble(cell(f, iFeat[k]));
            }
            e.servedScore = toDouble(cell(f, iScore));
            e.retrievalSources = cell(f, iSrc);
            e.watchRatio = oit->second.watchRatio;
            e.completed = oit->second.completed;
            e.liked = oit->second.liked;
            e.shared = oit->second.shared;
            e.followed = oit->second.followed;
            e.notInterested = oit->second.notInterested;
            e.sessionExit = oit->second.sessionExit;
            if (withSurvey) {
                if (auto sit = survey.find(key); sit != survey.end()) {
                    e.hasSurvey = true;
                    e.likert = sit->second;
                    ++ds.surveyRows;
                }
            }
            ds.rows.push_back(std::move(e));
        }
    }

    // Canonical, hash-map-order-independent row order so training is bit-reproducible.
    std::sort(ds.rows.begin(), ds.rows.end(), [](const Example &a, const Example &b) {
        return std::tie(a.requestId, a.reelId) < std::tie(b.requestId, b.reelId);
    });
    return ds;
}

// --- per-target train + evaluate ----------------------------------------------------------------

TargetResult trainAndEvaluateTarget(const Dataset &ds, const Split &split, const TargetSpec &target,
                                    const SgdHyperparams &hp, SplitMode mode) {
    TargetResult result;
    const std::string splitName(splitModeName(mode));

    // Gather applicable train/test rows (satisfaction restricts to survey rows).
    std::vector<std::size_t> trIdx;
    std::vector<std::size_t> teIdx;
    std::vector<double> yTrain;
    std::vector<double> yTest;
    for (std::size_t i : split.train) {
        if (auto v = extractTarget(ds.rows[i], target)) {
            trIdx.push_back(i);
            yTrain.push_back(*v);
        }
    }
    for (std::size_t i : split.test) {
        if (auto v = extractTarget(ds.rows[i], target)) {
            teIdx.push_back(i);
            yTest.push_back(*v);
        }
    }

    // Honest SKIP (contracts §5/§7): rare-follow / sparse-survey path.
    if (target.kind == TargetKind::Binary) {
        int posTrain = 0;
        for (double y : yTrain) {
            posTrain += (y > 0.5) ? 1 : 0;
        }
        if (trIdx.empty() || teIdx.empty() || posTrain < kMinPositivesToTrain) {
            result.skipped = true;
            result.skipReason = "only " + std::to_string(posTrain) + " positives in train (<" +
                                std::to_string(kMinPositivesToTrain) + ") or empty split";
            return result;
        }
    } else {
        if (static_cast<int>(trIdx.size()) < kMinPositivesToTrain || teIdx.empty()) {
            result.skipReason = "only " + std::to_string(trIdx.size()) + " train examples (<" +
                                std::to_string(kMinPositivesToTrain) + ") or empty test split";
            result.skipped = true;
            return result;
        }
    }

    std::vector<std::vector<double>> xTrain;
    xTrain.reserve(trIdx.size());
    for (std::size_t i : trIdx) {
        xTrain.push_back(featuresOf(ds.rows[i]));
    }
    std::vector<std::vector<double>> xTest;
    xTest.reserve(teIdx.size());
    for (std::size_t i : teIdx) {
        xTest.push_back(featuresOf(ds.rows[i]));
    }

    const double baseRate = meanOf(yTrain);
    const int nTr = static_cast<int>(trIdx.size());
    const int nTe = static_cast<int>(teIdx.size());

    // Learned predictions.
    std::vector<double> predLearned(teIdx.size());
    if (target.kind == TargetKind::Binary) {
        LogisticRegression lr;
        lr.train(xTrain, yTrain, hp, featureColumnNames(), target.name);
        for (std::size_t t = 0; t < xTest.size(); ++t) {
            predLearned[t] = lr.predictProba(xTest[t]);
        }
        result.modelJson = lr.toJson().dump(2);
    } else {
        LinearRegression lr;
        lr.train(xTrain, yTrain, hp, featureColumnNames(), target.name);
        for (std::size_t t = 0; t < xTest.size(); ++t) {
            predLearned[t] = lr.predict(xTest[t]);
        }
        result.modelJson = lr.toJson().dump(2);
    }

    // Baseline predictions on the test split.
    const std::vector<double> predGlobal(teIdx.size(), baseRate);

    std::unordered_map<std::string, std::pair<double, int>> srcAgg; // key -> (sum y, count)
    for (std::size_t t = 0; t < trIdx.size(); ++t) {
        auto &agg = srcAgg[majoritySourceKey(ds.rows[trIdx[t]].retrievalSources)];
        agg.first += yTrain[t];
        agg.second += 1;
    }
    std::vector<double> predPerSource(teIdx.size());
    for (std::size_t t = 0; t < teIdx.size(); ++t) {
        auto it = srcAgg.find(majoritySourceKey(ds.rows[teIdx[t]].retrievalSources));
        predPerSource[t] = (it != srcAgg.end() && it->second.second > 0)
                               ? it->second.first / static_cast<double>(it->second.second)
                               : baseRate;
    }

    std::vector<double> predServed(teIdx.size());
    for (std::size_t t = 0; t < teIdx.size(); ++t) {
        predServed[t] = ds.rows[teIdx[t]].servedScore;
    }

    result.evalRows.push_back(makeRow(target.name, "learned", splitName, nTr, nTe, predLearned,
                                      yTest, target.kind, true, baseRate));
    result.evalRows.push_back(makeRow(target.name, "global_frequency", splitName, nTr, nTe,
                                      predGlobal, yTest, target.kind, true, baseRate));
    result.evalRows.push_back(makeRow(target.name, "per_source_frequency", splitName, nTr, nTe,
                                      predPerSource, yTest, target.kind, true, baseRate));
    result.evalRows.push_back(makeRow(target.name, "served_score", splitName, nTr, nTe, predServed,
                                      yTest, target.kind, false, baseRate));

    result.calibration = equalCountBins(predLearned, yTest, 10);
    return result;
}

} // namespace rr::learning_v2
