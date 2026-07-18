#pragma once

#include <vector>

#include "rr/domain/reel.hpp"
#include "rr/infrastructure/config.hpp"
#include "rr/infrastructure/random.hpp"
#include "rr/simulation/hidden/hidden_reel_state.hpp"
#include "rr/simulation/modality_space.hpp"

namespace rr {

// Realism V2 reel-side augmentation (V2 TDD 4.1/4.4, Phase 13). Called by generateDataset ONLY
// when realism.content_v2 is on, AFTER the V1 generators have run — the V1 reel fields are
// produced by the untouched V1 path, so gate-on leaves them byte-identical and gate-off performs
// zero V2 draws (D17). Both functions consume caller-forked streams (D19): `reelsV2Rng` is the
// "reels-v2" fork, `archetypesRng` the "archetypes" fork; neither function calls forkRng.
//
// PACKAGE-A OWNERSHIP, FROZEN SIGNATURES: package A implements these in reel_augmenter_v2.cpp
// (currently a scaffolding stub). The signatures below must not change — dataset_generator.cpp
// (Fable-owned) calls them in this exact order.

// Generate the shared modality style centres as the FIRST draws on "reels-v2" (pinned order:
// centres before any per-reel draw, so widening simulation.reels never changes the centres, and
// user-side preferences built from the same centres stay stable under reel-count changes).
// Centre counts are named constants in the .cpp (D24); every centre is L2-normalized with
// dimension config.dimensions.
ModalitySpaces generateModalitySpaces(const SimulationConfig &config, const RealismConfig &realism,
                                      Rng &reelsV2Rng);

// Fill every reel's V2 serving-visible attributes and the index-aligned HiddenReelState vector:
// per reel, sample the archetype from `archetypesRng` (weighted by the catalog's mixture
// weights), then draw the attribute values conditionally from `reelsV2Rng` — archetypes shape
// the distributions probabilistically; no label reaches recommender-visible state (D18). The
// per-reel draw COUNT on each stream must be fixed and documented at the definition (independent
// of which archetype was sampled), so catalog edits change values, never stream alignment.
// Modality embeddings are L2-normalized centre blends; scalars are clamped to [0, 1]; language
// is drawn from rr::languageWeights(realism.languages).
void augmentReelsV2(std::vector<Reel> &reels, std::vector<HiddenReelState> &hiddenReelStates,
                    const ModalitySpaces &spaces, const SimulationConfig &config,
                    const RealismConfig &realism, Rng &archetypesRng, Rng &reelsV2Rng);

} // namespace rr
