#include "rr/simulation/reel_augmenter_v2.hpp"

#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "rr/core/embedding.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/simulation/dataset_generator.hpp"
#include "rr/simulation/language.hpp"

namespace {

// The modality style-centre counts are named constants inside reel_augmenter_v2.cpp (D24, not
// exposed on any surface). These mirrors let the tests pin the documented contract; update both
// together if the constants ever change.
constexpr std::size_t kVisualCentres = 12;
constexpr std::size_t kMusicCentres = 16;
constexpr std::size_t kEmotionalCentres = 8;

rr::SimulationConfig smallSim() {
    rr::SimulationConfig c;
    c.reels = 400;
    c.users = 40;
    c.creators = 80;
    c.topics = 16;
    c.dimensions = 32;
    return c;
}

rr::RealismConfig realismOn() {
    rr::RealismConfig r;
    r.contentV2 = true;
    r.languages = 8; // default catalog
    return r;
}

// Index of the archetype with the given name in a catalog.
std::uint32_t archetypeIndex(const std::vector<rr::ArchetypeSpec> &catalog, const std::string &n) {
    for (std::uint32_t i = 0; i < catalog.size(); ++i) {
        if (catalog[i].name == n) {
            return i;
        }
    }
    ADD_FAILURE() << "archetype not found: " << n;
    return 0;
}

double pearson(const std::vector<double> &x, const std::vector<double> &y) {
    const std::size_t n = x.size();
    const double mx = std::accumulate(x.begin(), x.end(), 0.0) / static_cast<double>(n);
    const double my = std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(n);
    double sxy = 0.0, sxx = 0.0, syy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = x[i] - mx;
        const double dy = y[i] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    return sxy / std::sqrt(sxx * syy);
}

} // namespace

// --- Validity and structure (gate on) --------------------------------------------------------

TEST(ReelAugmenterV2Test, ModalitySpacesHaveDocumentedCentreCountsAndAreUnitVectors) {
    const auto sim = smallSim();
    const auto ds = rr::generateDataset(sim, realismOn(), 7u);
    EXPECT_EQ(ds.modalitySpaces.visualCentres.size(), kVisualCentres);
    EXPECT_EQ(ds.modalitySpaces.musicCentres.size(), kMusicCentres);
    EXPECT_EQ(ds.modalitySpaces.emotionalCentres.size(), kEmotionalCentres);
    for (const auto *group : {&ds.modalitySpaces.visualCentres, &ds.modalitySpaces.musicCentres,
                              &ds.modalitySpaces.emotionalCentres}) {
        for (const rr::Embedding &centre : *group) {
            ASSERT_EQ(centre.size(), sim.dimensions);
            EXPECT_TRUE(rr::isValid(centre)); // non-empty, finite, unit-length
        }
    }
}

TEST(ReelAugmenterV2Test, EveryReelAttributeIsValid) {
    const auto sim = smallSim();
    const auto realism = realismOn();
    const auto ds = rr::generateDataset(sim, realism, 11u);

    ASSERT_EQ(ds.reels.size(), sim.reels);
    for (const rr::Reel &r : ds.reels) {
        // Modality embeddings: unit-norm, config dimension.
        for (const rr::Embedding *e :
             {&r.visualStyleEmbedding, &r.musicEmbedding, &r.emotionalToneEmbedding}) {
            ASSERT_EQ(e->size(), sim.dimensions);
            EXPECT_TRUE(rr::isValid(*e));
        }
        // Scalars: all in [0, 1].
        for (float s : {r.usefulness, r.humour, r.novelty, r.productionQuality, r.controversy,
                        r.clickbaitStrength, r.informationDensity, r.emotionalIntensity}) {
            EXPECT_GE(s, 0.0f);
            EXPECT_LE(s, 1.0f);
        }
        // Language id within the configured set.
        EXPECT_LT(r.language.value, realism.languages);
    }
}

TEST(ReelAugmenterV2Test, HiddenReelStatesAreAlignedSizedAndValid) {
    const auto sim = smallSim();
    const auto realism = realismOn();
    const auto ds = rr::generateDataset(sim, realism, 13u);

    ASSERT_EQ(ds.hiddenReelStates.size(), ds.reels.size());
    for (std::size_t i = 0; i < ds.reels.size(); ++i) {
        const rr::HiddenReelState &h = ds.hiddenReelStates[i];
        EXPECT_EQ(h.reelId, ds.reels[i].id); // index-aligned with reels
        EXPECT_LT(h.archetypeIndex, realism.archetypes.size());
        // niche cohort centre lives in the hash01 [0, 1) space and is 0 for non-niche reels.
        EXPECT_GE(h.nicheCohortCentre, 0.0f);
        EXPECT_LT(h.nicheCohortCentre, 1.0f);
        if (h.nicheCohortWidth <= 0.0f) {
            EXPECT_EQ(h.nicheCohortCentre, 0.0f);
        }
    }
}

// D17 / "zero V1-field mutation": generating with the gate ON must leave every V1 reel field
// byte-identical to the gate-OFF run (same seed) — the V1 generators run first and untouched, and
// augmentReelsV2 only WRITES V2 fields (it reads intrinsicQuality, never writes it).
TEST(ReelAugmenterV2Test, GateOnLeavesV1ReelFieldsIdentical) {
    const auto sim = smallSim();
    rr::RealismConfig off; // contentV2 = false
    const auto dsOff = rr::generateDataset(sim, off, 21u);
    const auto dsOn = rr::generateDataset(sim, realismOn(), 21u);

    // Gate-off performs zero V2 augmentation (D17).
    EXPECT_TRUE(dsOff.hiddenReelStates.empty());
    EXPECT_TRUE(dsOff.modalitySpaces.visualCentres.empty());

    ASSERT_EQ(dsOff.reels.size(), dsOn.reels.size());
    for (std::size_t i = 0; i < dsOff.reels.size(); ++i) {
        const rr::Reel &a = dsOff.reels[i];
        const rr::Reel &b = dsOn.reels[i];
        EXPECT_EQ(a.id, b.id);
        EXPECT_EQ(a.creatorId, b.creatorId);
        EXPECT_EQ(a.embedding, b.embedding); // semantic (ANN-indexed) vector untouched
        EXPECT_FLOAT_EQ(a.intrinsicQuality, b.intrinsicQuality);
        EXPECT_FLOAT_EQ(a.durationSeconds, b.durationSeconds);
        EXPECT_EQ(a.primaryTopic, b.primaryTopic);
        EXPECT_EQ(a.secondaryTopics, b.secondaryTopics);
        EXPECT_EQ(a.createdAt, b.createdAt);
        EXPECT_EQ(a.active, b.active);
    }
}

// productionQuality is drawn CORRELATED with the V1 intrinsicQuality (plus an archetype shift and
// noise), so across the population the two must be clearly positively correlated while remaining
// distinct (reel.hpp reconciliation).
TEST(ReelAugmenterV2Test, ProductionQualityCorrelatesWithIntrinsicQuality) {
    rr::SimulationConfig sim = smallSim();
    sim.reels = 6000;
    const auto ds = rr::generateDataset(sim, realismOn(), 23u);
    std::vector<double> iq, pq;
    iq.reserve(ds.reels.size());
    pq.reserve(ds.reels.size());
    for (const rr::Reel &r : ds.reels) {
        iq.push_back(r.intrinsicQuality);
        pq.push_back(r.productionQuality);
    }
    const double corr = pearson(iq, pq);
    EXPECT_GT(corr, 0.2) << "productionQuality/intrinsicQuality correlation too weak: " << corr;
    EXPECT_LT(corr, 0.999) << "productionQuality must not be aliased to intrinsicQuality";
}

TEST(ReelAugmenterV2Test, SameSeedProducesIdenticalAugmentation) {
    const auto sim = smallSim();
    const auto realism = realismOn();
    const auto a = rr::generateDataset(sim, realism, 31u);
    const auto b = rr::generateDataset(sim, realism, 31u);
    ASSERT_EQ(a.reels.size(), b.reels.size());
    for (std::size_t i = 0; i < a.reels.size(); ++i) {
        EXPECT_EQ(a.reels[i].visualStyleEmbedding, b.reels[i].visualStyleEmbedding);
        EXPECT_EQ(a.reels[i].musicEmbedding, b.reels[i].musicEmbedding);
        EXPECT_EQ(a.reels[i].emotionalToneEmbedding, b.reels[i].emotionalToneEmbedding);
        EXPECT_FLOAT_EQ(a.reels[i].usefulness, b.reels[i].usefulness);
        EXPECT_FLOAT_EQ(a.reels[i].productionQuality, b.reels[i].productionQuality);
        EXPECT_EQ(a.reels[i].language.value, b.reels[i].language.value);
        EXPECT_EQ(a.hiddenReelStates[i].archetypeIndex, b.hiddenReelStates[i].archetypeIndex);
        EXPECT_FLOAT_EQ(a.hiddenReelStates[i].nicheCohortCentre,
                        b.hiddenReelStates[i].nicheCohortCentre);
    }
}

// --- Statistical suite: one documented seed, a shared 50k-reel dataset, generous margins ------

class ReelAugmenterStats : public ::testing::Test {
  protected:
    static rr::GeneratedDataset ds_;
    static rr::RealismConfig realism_;
    static std::vector<double> archetypeMean_; // proportion counts helper reused across tests

    static void SetUpTestSuite() {
        rr::SimulationConfig c;
        c.reels = 50000;
        c.users = 100;
        c.creators = 1000;
        c.topics = 32;
        c.dimensions = 32;
        realism_.contentV2 = true;
        realism_.languages = 8;
        ds_ = rr::generateDataset(c, realism_, /*documented seed=*/2026u);
    }
    static void TearDownTestSuite() { ds_ = rr::GeneratedDataset{}; }

    // Mean of an attribute over reels whose archetype index equals `arch` (or over all reels when
    // arch < 0). `get` extracts the scalar from a reel.
    template <class F> double meanOver(int arch, F get) const {
        double sum = 0.0;
        std::size_t n = 0;
        for (std::size_t i = 0; i < ds_.reels.size(); ++i) {
            if (arch < 0 ||
                ds_.hiddenReelStates[i].archetypeIndex == static_cast<std::uint32_t>(arch)) {
                sum += get(ds_.reels[i]);
                ++n;
            }
        }
        return n == 0 ? 0.0 : sum / static_cast<double>(n);
    }
};

rr::GeneratedDataset ReelAugmenterStats::ds_;
rr::RealismConfig ReelAugmenterStats::realism_;
std::vector<double> ReelAugmenterStats::archetypeMean_;

TEST_F(ReelAugmenterStats, MixtureProportionsMatchConfiguredWeights) {
    const auto &catalog = realism_.archetypes;
    double totalW = 0.0;
    for (const auto &a : catalog) {
        totalW += a.weight;
    }
    std::vector<std::size_t> counts(catalog.size(), 0);
    for (const auto &h : ds_.hiddenReelStates) {
        ++counts[h.archetypeIndex];
    }
    const double n = static_cast<double>(ds_.reels.size());
    for (std::size_t a = 0; a < catalog.size(); ++a) {
        const double observed = counts[a] / n;
        const double expected = catalog[a].weight / totalW;
        // ~50k reels => sampling std ~0.002; a max(0.01, 10% relative) band is many sigma of slack.
        const double tol = std::max(0.01, 0.10 * expected);
        EXPECT_NEAR(observed, expected, tol)
            << catalog[a].name << " observed=" << observed << " expected=" << expected;
    }
}

TEST_F(ReelAugmenterStats, LanguageHistogramMatchesSkewedWeights) {
    const auto weights = rr::languageWeights(realism_.languages);
    std::vector<std::size_t> counts(realism_.languages, 0);
    for (const auto &r : ds_.reels) {
        ASSERT_LT(r.language.value, realism_.languages);
        ++counts[r.language.value];
    }
    const double n = static_cast<double>(ds_.reels.size());
    // Dominant language 0 is the most frequent, and the histogram is monotone-ish decreasing.
    EXPECT_EQ(std::distance(counts.begin(), std::max_element(counts.begin(), counts.end())), 0);
    for (std::uint32_t i = 0; i < realism_.languages; ++i) {
        const double observed = counts[i] / n;
        EXPECT_NEAR(observed, weights[i], std::max(0.01, 0.12 * weights[i]))
            << "language " << i << " observed=" << observed << " expected=" << weights[i];
    }
}

TEST_F(ReelAugmenterStats, ArchetypeSignaturesSeparateFromPopulation) {
    const auto &cat = realism_.archetypes;
    const int rage = static_cast<int>(archetypeIndex(cat, "ragebait"));
    const int click = static_cast<int>(archetypeIndex(cat, "clickbait"));
    const int useful = static_cast<int>(archetypeIndex(cat, "useful"));
    const int polished = static_cast<int>(archetypeIndex(cat, "polished_irrelevant"));

    const double popControversy = meanOver(-1, [](const rr::Reel &r) { return r.controversy; });
    const double popClickbait = meanOver(-1, [](const rr::Reel &r) { return r.clickbaitStrength; });
    const double popUsefulness = meanOver(-1, [](const rr::Reel &r) { return r.usefulness; });
    const double popProduction =
        meanOver(-1, [](const rr::Reel &r) { return r.productionQuality; });

    const double rageControversy = meanOver(rage, [](const rr::Reel &r) { return r.controversy; });
    const double clickStrength =
        meanOver(click, [](const rr::Reel &r) { return r.clickbaitStrength; });
    const double usefulUsefulness =
        meanOver(useful, [](const rr::Reel &r) { return r.usefulness; });
    const double polishedProduction =
        meanOver(polished, [](const rr::Reel &r) { return r.productionQuality; });

    // Each designed signal ≫ the population mean (large, generous margins).
    EXPECT_GT(rageControversy, popControversy + 0.25)
        << "ragebait controversy=" << rageControversy << " pop=" << popControversy;
    EXPECT_GT(clickStrength, popClickbait + 0.25)
        << "clickbait strength=" << clickStrength << " pop=" << popClickbait;
    EXPECT_GT(usefulUsefulness, popUsefulness + 0.2)
        << "useful usefulness=" << usefulUsefulness << " pop=" << popUsefulness;
    EXPECT_GT(polishedProduction, popProduction + 0.1)
        << "polished production=" << polishedProduction << " pop=" << popProduction;
}

TEST_F(ReelAugmenterStats, BackgroundMusicModalityCoherenceIsTighter) {
    const int music = static_cast<int>(archetypeIndex(realism_.archetypes, "background_music"));
    const auto &centres = ds_.modalitySpaces.musicCentres;
    ASSERT_FALSE(centres.empty());

    auto nearestCosine = [](const rr::Embedding &e, const std::vector<rr::Embedding> &cs) {
        double best = -2.0;
        for (const rr::Embedding &c : cs) {
            best = std::max(best, static_cast<double>(rr::dot(e, c)));
        }
        return best;
    };

    double bgSum = 0.0, popSum = 0.0;
    std::size_t bgN = 0;
    for (std::size_t i = 0; i < ds_.reels.size(); ++i) {
        const double cos = nearestCosine(ds_.reels[i].musicEmbedding, centres);
        popSum += cos;
        if (ds_.hiddenReelStates[i].archetypeIndex == static_cast<std::uint32_t>(music)) {
            bgSum += cos;
            ++bgN;
        }
    }
    ASSERT_GT(bgN, 0u);
    const double bgMean = bgSum / static_cast<double>(bgN);
    const double popMean = popSum / static_cast<double>(ds_.reels.size());
    EXPECT_GT(bgMean, popMean + 0.05)
        << "background_music music-coherence not tighter: bg=" << bgMean << " pop=" << popMean;
    EXPECT_GT(bgMean, 0.9) << "background_music music embeddings should hug their centre";
}

// The hidden per-reel state must faithfully carry the sampled archetype's hidden parameters
// (satisfaction/regret/hook/decay/comfort/niche), and the niche cohort centre must vary per reel.
TEST_F(ReelAugmenterStats, HiddenStateSignaturesMatchArchetypeDesign) {
    const auto &cat = realism_.archetypes;
    const auto idx = [&](const std::string &n) { return archetypeIndex(cat, n); };

    std::size_t rageN = 0, clickN = 0, comfortN = 0, nicheN = 0;
    float nicheMin = 2.0f, nicheMax = -1.0f;
    for (const auto &h : ds_.hiddenReelStates) {
        if (h.archetypeIndex == idx("ragebait")) {
            ++rageN;
            EXPECT_LT(h.satisfactionBias, 0.0f);
            EXPECT_GT(h.regretBias, 0.0f);
        } else if (h.archetypeIndex == idx("clickbait")) {
            ++clickN;
            EXPECT_GT(h.openingHook, 0.0f);
            EXPECT_GT(h.retentionDecay, 0.0f);
        } else if (h.archetypeIndex == idx("comfort")) {
            ++comfortN;
            EXPECT_GT(h.comfortReturnBonus, 0.0f);
        } else if (h.archetypeIndex == idx("niche_treasure")) {
            ++nicheN;
            EXPECT_GT(h.nicheCohortWidth, 0.0f);
            nicheMin = std::min(nicheMin, h.nicheCohortCentre);
            nicheMax = std::max(nicheMax, h.nicheCohortCentre);
        }
    }
    EXPECT_GT(rageN, 0u);
    EXPECT_GT(clickN, 0u);
    EXPECT_GT(comfortN, 0u);
    ASSERT_GT(nicheN, 0u);
    // Niche cohort centres are drawn per reel across the whole [0, 1) hash01 space.
    EXPECT_GT(nicheMax - nicheMin, 0.5f) << "niche cohort centres should vary per reel";
}
