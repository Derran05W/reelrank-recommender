#pragma once

#include <vector>

#include "rr/core/embedding.hpp"

namespace rr {

// Shared latent structure for the Realism V2 modality channels (V2 TDD 4.1/4.2, Phase 13): a
// small set of L2-normalized style centres per modality, generated once per dataset by
// generateModalitySpaces as the FIRST draws on stream "reels-v2" (pinned order — centres before
// any per-reel draw, so widening simulation.reels never changes the centres). Reel modality
// embeddings blend these centres (archetype-conditioned) and hidden user modality preferences
// blend the SAME centres, giving the channels learnable shared structure — without a shared
// basis, modality dot products would be structureless noise.
//
// The centres are content structure, not user truth: they carry no hidden user information and
// need no D18 isolation (the reel modality embeddings built from them are serving-visible
// anyway). Centre counts are named constants in reel_augmenter_v2.cpp (D24 no-premature-config).
struct ModalitySpaces {
    std::vector<Embedding> visualCentres;
    std::vector<Embedding> musicCentres;
    std::vector<Embedding> emotionalCentres;
};

} // namespace rr
