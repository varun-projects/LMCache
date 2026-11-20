# LMCache — an engineer's guide

A from-scratch technical guide to the [LMCache](https://github.com/LMCache/LMCache)
codebase: what it is, how it is put together, how each major subsystem works, and
how five real bugs were diagnosed and fixed.

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

How five real, **already-fixed** bugs were found and solved — the most useful way to
learn a codebase. Each walks the symptom, the root cause, the actual merged diff, and
why the fix is correct.

See the [deep dive index](issue-deep-dives/README.md).

| Difficulty | Issue | One-line summary |
|---|---|---|
| **Difficult** | [#5151](issue-deep-dives/difficult-5151-mamba-full-hit-restoration.md) | Lookup reported a full hit but restored the wrong KV for long prefixes — broke somewhere between 114 and 128 chunks |
| **Difficult** | [#5022](issue-deep-dives/difficult-5022-turboquant-sm80-bfloat16.md) | A TurboQuant Triton kernel asserted on A100 (SM80) for bfloat16 models |
| Medium | [#4416](issue-deep-dives/medium-4416-percentile-off-by-one.md) | Trace-replay p90/p99 reported the maximum, via an off-by-one at whole ranks |
| Medium | [#4591](issue-deep-dives/medium-4591-sdk-output-tokens.md) | `LMCacheRequestStream.output_tokens` counted the prompt as output |
| Medium | [#4698](issue-deep-dives/medium-4698-nixl-pool-slot-units.md) | A NIXL L2 adapter reported *negative* pool usage |

## Conventions used throughout

- **Flowcharts are [Mermaid](https://mermaid.js.org/)**, which GitHub renders natively in Markdown. If you are reading this outside GitHub, each diagram is followed by prose that carries the same information.
- **Code links are permalinks** pinned to `53bde913`, so they will not rot.
- **Claims are separated from inference.** Where a statement comes from reading code it is stated as such; where something could not be verified (no GPU, no multi-node cluster) that is said explicitly rather than glossed over.
- `docs/design/` in the upstream repo **mirrors the `lmcache/` package tree** — a design doc for `lmcache/<path>/` lives at `docs/design/<path>/`. These documents cite those design docs where they exist and are still accurate.

## What this guide is not

- Not a replacement for the upstream [user documentation](https://docs.lmcache.ai/) — that covers installation, configuration reference and recipes.
- Not an API reference. It explains structure and reasoning, not every signature.
- Not a performance study. No benchmarks were run for this guide, and it makes no throughput or latency claims of its own.
