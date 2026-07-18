#include "rr/simulation/dataset_generator.hpp"

#include <utility>

#include "rr/infrastructure/random.hpp"
#include "rr/simulation/creator_generator.hpp"
#include "rr/simulation/reel_augmenter_v2.hpp"
#include "rr/simulation/reel_generator.hpp"
#include "rr/simulation/topic_generator.hpp"
#include "rr/simulation/user_augmenter_v2.hpp"
#include "rr/simulation/user_generator.hpp"

namespace rr {

GeneratedDataset generateDataset(const SimulationConfig &config, uint64_t seed) {
    return generateDataset(config, RealismConfig{}, seed);
}

GeneratedDataset generateDataset(const SimulationConfig &config, const RealismConfig &realism,
                                 uint64_t seed) {
    GeneratedDataset dataset;

    Rng topicsRng = forkRng(seed, "topics");
    dataset.topics = generateTopics(config, topicsRng);

    Rng creatorsRng = forkRng(seed, "creators");
    dataset.creators = generateCreators(config, dataset.topics, creatorsRng);

    Rng reelsRng = forkRng(seed, "reels");
    dataset.reels = generateReels(config, dataset.topics, dataset.creators, reelsRng);

    Rng usersRng = forkRng(seed, "users");
    GeneratedUsers generatedUsers = generateUsers(config, dataset.topics, usersRng);
    dataset.users = std::move(generatedUsers.users);
    dataset.hiddenStates = std::move(generatedUsers.hiddenStates);

    // Realism V2 augmentation (Phase 13, D17/D19): runs strictly AFTER the V1 generators so the
    // V1 fields above are produced by the untouched V1 path, and only under the gate — gate-off
    // forks none of the V2 streams and performs zero V2 draws. Pinned order: modality spaces
    // (first draws on "reels-v2"), then reel augmentation ("archetypes" + "reels-v2"), then user
    // augmentation ("users-v2", consuming the modality spaces as data).
    if (realism.contentV2) {
        Rng archetypesRng = forkRng(seed, "archetypes");
        Rng reelsV2Rng = forkRng(seed, "reels-v2");
        Rng usersV2Rng = forkRng(seed, "users-v2");
        dataset.modalitySpaces = generateModalitySpaces(config, realism, reelsV2Rng);
        augmentReelsV2(dataset.reels, dataset.hiddenReelStates, dataset.modalitySpaces, config,
                       realism, archetypesRng, reelsV2Rng);
        augmentUsersV2(dataset.hiddenStates, dataset.modalitySpaces, config, realism, usersV2Rng);
    }

    return dataset;
}

std::size_t appendUsers(GeneratedDataset &ds, const SimulationConfig &config, uint64_t masterSeed,
                        uint32_t count) {
    const std::size_t firstNewIndex = ds.users.size();
    if (count == 0) {
        return firstNewIndex;
    }

    // Generate exactly `count` users on the dedicated "users-injected" stream (D8), id-shifted so
    // they continue the dense sequence. A config copy carries only the injected count; dimensions
    // and topics come from the live dataset, so injected users mix the SAME topic space.
    SimulationConfig cfg = config;
    cfg.users = count;
    Rng rng = forkRng(masterSeed, "users-injected");
    GeneratedUsers gen = generateUsers(cfg, ds.topics, rng, static_cast<uint32_t>(firstNewIndex));

    ds.users.reserve(ds.users.size() + gen.users.size());
    ds.hiddenStates.reserve(ds.hiddenStates.size() + gen.hiddenStates.size());
    for (User &u : gen.users) {
        ds.users.push_back(std::move(u));
    }
    for (HiddenUserState &h : gen.hiddenStates) {
        ds.hiddenStates.push_back(std::move(h));
    }
    return firstNewIndex;
}

std::size_t appendReels(GeneratedDataset &ds, const SimulationConfig &config, uint64_t masterSeed,
                        uint32_t count, Timestamp createdAt) {
    const std::size_t firstNewIndex = ds.reels.size();
    if (count == 0) {
        return firstNewIndex;
    }

    // Generate exactly `count` reels on the dedicated "reels-injected" stream (D8), id-shifted to
    // continue the dense sequence, over the EXISTING topics/creators (the normal reel machinery).
    SimulationConfig cfg = config;
    cfg.reels = count;
    Rng rng = forkRng(masterSeed, "reels-injected");
    std::vector<Reel> fresh =
        generateReels(cfg, ds.topics, ds.creators, rng, static_cast<uint32_t>(firstNewIndex));

    ds.reels.reserve(ds.reels.size() + fresh.size());
    for (Reel &r : fresh) {
        // These ARE the fresh content: stamp createdAt to the injection-time clock (D9), overriding
        // the generator's random-window draw. Counters are already zero and active already true.
        r.createdAt = createdAt;
        ds.reels.push_back(std::move(r));
    }
    return firstNewIndex;
}

} // namespace rr
