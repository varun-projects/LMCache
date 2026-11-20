# LMCache — an engineer's guide

A from-scratch technical guide to the [LMCache](https://github.com/LMCache/LMCache)
codebase: what it is, how it is put together, how each major subsystem works, how ten
real bugs were diagnosed and fixed, and how three merged PRs made the design calls they
did.

Written for someone who has just cloned the repository and wants a mental model
before touching code.

---

## Which revision this describes

> **All code references, paths, line numbers and permalinks in this guide point at
> upstream `dev` @ [`53bde913`](https://github.com/LMCache/LMCache/tree/53bde913de1adae2a380fe0c2de8d2b9355b4702)**
> — the revision that was analysed. It was the tip of `LMCache/LMCache:dev` when this
> guide was started; `dev` has since moved on, which is exactly why every link below
> is pinned to a commit rather than to a branch.
>
> This `intro` branch is based on an older fork `dev` (`ee68e5ba`) purely so that it
> could be published without a `workflow`-scoped token. **Do not read the code in
> this branch's tree as the subject of these documents** — follow the permalinks,
> which are pinned to `53bde913` and will stay correct.
>
> LMCache moves fast (≈500 open PRs at time of writing). Treat every line number as
> a pointer that may drift, and the named symbols as the durable part.

## Start here

| # | Document | What it answers |
|---|---|---|
| 1 | [Project overview](01-project-overview.md) | What problem LMCache solves, where it sits relative to vLLM/SGLang, and the vocabulary you need |
| 2 | [Architecture](02-architecture.md) | The process topology, the storage hierarchy, and how a request flows end to end |
| 3 | [Code map](03-code-map.md) | What lives where in the repo, and how to find the code behind a behaviour |

## Subsystem deep dives

One document per major feature. Each is self-contained and starts from its public
entry point.

| Document | Subsystem |
|---|---|
| [Multiprocess (MP) architecture](features/01-mp-architecture.md) | The MP server, coordinator and module system — the backbone of modern LMCache |
| [KV cache lifecycle](features/02-kv-cache-lifecycle.md) | `lookup` → `prefetch` → `retrieve` → `store`, and how chunks are keyed |
| [L1 memory and eviction](features/03-l1-memory-and-eviction.md) | The CPU/GPU memory pool, allocation, pinning, ref counts and eviction policies |
| [L2 adapters and storage tiers](features/04-l2-adapters-and-storage.md) | The pluggable L2 backends (fs, NIXL, Valkey, Mooncake, RawBlock, …) |
| [vLLM integration](features/05-vllm-integration.md) | The KV connector, scheduler hooks, KV cache groups and hybrid models |
| [Transport layer](features/06-transport-layer.md) | gRPC, ZMQ, shared memory and NIXL/UCX data paths |
| [Serde and compression](features/07-serde-and-compression.md) | Serialization, CacheGen, TurboQuant and encryption wrappers |
| [P2P cache sharing](features/08-p2p-sharing.md) | Reading KV directly from a peer node's L1 |
| [Observability](features/09-observability.md) | The event bus, metrics, tracing and the log surface |
| [CLI](features/10-cli.md) | `lmcache server`, `bench`, `query`, `trace` and the command framework |

## Issue deep dives

How ten real, **already-fixed** bugs were found and solved — the most useful way to
learn a codebase. Each walks the symptom, the root cause, the actual merged diff, and
why the fix is correct.

See the [deep dive index](issue-deep-dives/README.md) for reading order, recurring themes
and a per-document verification summary.

| Difficulty | Issue | One-line summary |
|---|---|---|
| **Difficult** | [#5151](issue-deep-dives/difficult-5151-mamba-full-hit-restoration.md) | Lookup reported a full hit but restored the wrong KV for long prefixes — broke somewhere between 114 and 128 chunks |
| **Difficult** | [#5713](issue-deep-dives/difficult-5713-lora-cache-key-collision.md) | Every LoRA adapter shared the base model's cache keys, so requests were served KV computed under different weights |
| **Difficult** | [#6392](issue-deep-dives/difficult-6392-prefetch-l1-landing-race.md) | A chunk that landed in L1 during the L2 lookup became a miss, and one dropped chunk truncated the whole prefix |
| **Difficult** | [#5022](issue-deep-dives/difficult-5022-turboquant-sm80-bfloat16.md) | A TurboQuant Triton kernel asserted on A100 (SM80) for bfloat16 models |
| Medium | [#4416](issue-deep-dives/medium-4416-percentile-off-by-one.md) | Trace-replay p90/p99 reported the maximum, via an off-by-one at whole ranks |
| Medium | [#4591](issue-deep-dives/medium-4591-sdk-output-tokens.md) | `LMCacheRequestStream.output_tokens` counted the prompt as output |
| Medium | [#4698](issue-deep-dives/medium-4698-nixl-pool-slot-units.md) | A NIXL L2 adapter reported *negative* pool usage |
| Medium | [#5115](issue-deep-dives/medium-5115-address-manager-zero-size.md) | A zero-byte allocation fractured the free list — and the first fix turned that into a cache wipe |
| Medium | [#6543](issue-deep-dives/medium-6543-zmq-grpc-fail-fast.md) | A ZMQ client pointed at a gRPC server hung for 300 s, then blamed the wrong thing |
| Medium | [#6698](issue-deep-dives/medium-6698-blend-matcher-slot-ownership.md) | Evicting one copy of a chunk permanently hid another copy that was still stored |

## PR walkthroughs

Three merged PRs read as **design decisions** rather than bug hunts: what the options were,
and what the author traded. See the [walkthrough index](pr-walkthroughs/README.md).

| PR | Area | The decision |
|---|---|---|
| [#6767](pr-walkthroughs/6767-blend-chunk-fingerprinting.md) | CacheBlend × Mamba | Who belongs in an `all(...)` quorum once a member abstains by design |
| [#6754](pr-walkthroughs/6754-grpc-auto-server-mode.md) | gRPC transport | Changing a default to "the best thing installed", without weakening explicit values |
| [#6752](pr-walkthroughs/6752-ci-stale-nightly-provenance.md) | CI / release engineering | A green nightly run that proved nothing for two months |

## Conventions used throughout

- **Flowcharts are [Mermaid](https://mermaid.js.org/)**, which GitHub renders natively in Markdown. If you are reading this outside GitHub, each diagram is followed by prose that carries the same information.
- **Code links are permalinks** pinned to `53bde913`, so they will not rot.
- **Claims are separated from inference.** Where a statement comes from reading code it is stated as such; where something could not be verified (no GPU, no multi-node cluster) that is said explicitly rather than glossed over.
- `docs/design/` in the upstream repo **mirrors the `lmcache/` package tree** — a design doc for `lmcache/<path>/` lives at `docs/design/<path>/`. These documents cite those design docs where they exist and are still accurate.

## What this guide is not

- Not a replacement for the upstream [user documentation](https://docs.lmcache.ai/) — that covers installation, configuration reference and recipes.
- Not an API reference. It explains structure and reasoning, not every signature.
- Not a performance study. No benchmarks were run for this guide, and it makes no throughput or latency claims of its own.
