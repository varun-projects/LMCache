# 3. Code map — what lives where

> Code references in this document are permalinks pinned to upstream `dev`
> @ [`53bde913`](https://github.com/LMCache/LMCache/tree/53bde913de1adae2a380fe0c2de8d2b9355b4702).

This document answers one question: *given a behaviour, which directory do I open?*
It is a map, not an explanation — each area links to the feature document that explains it.

---

## 3.1 Repository top level

| Path | What it is |
|---|---|
| `lmcache/` | The Python package. ~750 files. Everything below in §3.2. |
| `csrc/` | C++/CUDA extensions (89 files) — `lmcache_native`, CUDA/MUSA/SYCL kernels, storage backends, and the KV transfer plan headers. |
| `rust/` | Rust crates. Exactly one today: `raw_block`. |
| `tests/` | 605 files, mirroring the package layout (`tests/v1/...`). |
| `docs/source/` | User documentation, reStructuredText, Sphinx-built → [docs.lmcache.ai](https://docs.lmcache.ai/). |
| `docs/design/` | Design docs, Markdown, **mirroring the `lmcache/` tree**. See §3.5. |
| `benchmarks/`, `examples/` | Standalone scripts; not imported by the package. |
| `operator/` | Kubernetes operator. |
| `docker/`, `requirements/`, `setup_extensions/` | Packaging and build plumbing. |
| `AGENTS.md` / `CLAUDE.md` | Contributor and agent policy. `CLAUDE.md` is a symlink-equivalent copy; `AGENTS.md` is human-owned. |
| `.buildkite/` | CI pipelines, including the `k3_tests/multiprocess` suite that runs the real MP server. |

### A note on `v1`

Almost everything lives under `lmcache/v1/`. There is **no `v0`** in the tree — the prefix
is historical, from a rewrite whose predecessor was deleted rather than kept. Read `v1` as
"the engine", not as "one of several versions". The modules *outside* `v1/` are the ones
that sit above or beside the engine: `cli/`, `integration/`, `sdk/`, `security/`,
`usage_telemetry/`, `tools/`.

---

## 3.2 Inside `lmcache/`

```
lmcache/
├── v1/                  the engine
│   ├── distributed/     storage: L1, L2, serde, controllers      (108 files)
│   ├── multiprocess/    the MP server and its request modules     (89 files)
│   ├── platform/        device/backend abstraction                (79 files)
│   ├── mp_coordinator/  multi-server coordination                 (56 files)
│   ├── mp_observability/ event bus, metrics, tracing              (51 files)
│   ├── common/          shared types: keys, hashing, codecs       (50 files)
│   ├── transport/       gRPC, ZMQ, shared memory, RPC             (36 files)
│   ├── client/          client-side transport to the MP server    (21 files)
│   ├── gpu_connector/   GPU-side copy paths, GDS                  (13 files)
│   ├── memory_orchestrator/  cross-tier capacity arbitration      (12 files)
│   ├── memory_allocators/    host/device allocators               (11 files)
│   ├── kv_codec/        KV layout encode/decode
│   ├── plugin/          out-of-tree extension loading
│   └── kv_layer_groups.py, memory_management.py, protocol.py, …
├── cli/                 the `lmcache` command                    (102 files)
├── integration/         engine adapters: vllm, sglang, trtllm, atom  (37 files)
├── sdk/                 the direct-use Python SDK                 (11 files)
├── security/            API-key authentication
├── usage_telemetry/     opt-in usage reporting
├── tools/               cache simulator, transfer benchmarks, status viewer
├── lmcache_frontend/    a small web UI + MP plugin
└── logging.py, utils.py, connections.py, banner.py
```

---

## 3.3 "I want to change X" → open Y

### Storage and caching

| Behaviour | Where | Explained in |
|---|---|---|
| How a lookup/store/retrieve is orchestrated across tiers | [`v1/distributed/storage_manager.py`](https://github.com/LMCache/LMCache/blob/53bde913de1adae2a380fe0c2de8d2b9355b4702/lmcache/v1/distributed/storage_manager.py) | [KV cache lifecycle](features/02-kv-cache-lifecycle.md) |
| The data types everything passes around (`ObjectKey`, `MemoryLayoutDesc`, `MemoryTags`) | [`v1/distributed/api.py`](https://github.com/LMCache/LMCache/blob/53bde913de1adae2a380fe0c2de8d2b9355b4702/lmcache/v1/distributed/api.py) | [KV cache lifecycle](features/02-kv-cache-lifecycle.md) |
| Host-memory pool, pinning, ref counts, object state machine | `v1/distributed/l1_manager/` — one subclass per medium (`dram`, `devdax`, `gds`, `shadow`) | [L1 memory and eviction](features/03-l1-memory-and-eviction.md) |
| Eviction policy | `v1/distributed/eviction_policy/`, `eviction.py` | [L1 memory and eviction](features/03-l1-memory-and-eviction.md) |
| An external storage backend (S3, Valkey, Mooncake, NIXL, fs, …) | `v1/distributed/l2_adapters/` — ~25 adapters, one file each | [L2 adapters](features/04-l2-adapters-and-storage.md) |
| Background work on L1: prefetch, store, eviction, adapter lifecycle | `v1/distributed/storage_controllers/` | [L1 memory](features/03-l1-memory-and-eviction.md), [L2 adapters](features/04-l2-adapters-and-storage.md) |
| Compression, quantization, encryption | `v1/distributed/serde/` (`turboquant/`, `fp8.py`, `aesgcm.py`, `async_processor.py`) | [Serde and compression](features/07-serde-and-compression.md) |
| Per-tenant quotas | `v1/distributed/quota_manager.py` | [L2 adapters](features/04-l2-adapters-and-storage.md) |
| Reading KV from a peer node | `v1/distributed/p2p.py`, `l2_adapters/p2p_l2_adapter.py`, `multiprocess/modules/p2p_controller.py` | [P2P sharing](features/08-p2p-sharing.md) |

### The server

| Behaviour | Where | Explained in |
|---|---|---|
| Server startup, sessions, request dispatch, the module system | `v1/multiprocess/` | [MP architecture](features/01-mp-architecture.md) |
| A specific RPC (lookup, transfer, blend, kv events, management) | `v1/multiprocess/modules/` — one file per module | [MP architecture](features/01-mp-architecture.md) |
| Multi-server coordination, membership | `v1/mp_coordinator/` | [MP architecture](features/01-mp-architecture.md) |
| Wire protocol, gRPC/ZMQ servers, shared memory | `v1/transport/` | [Transport layer](features/06-transport-layer.md) |
| Client side of the same wire | `v1/client/` | [Transport layer](features/06-transport-layer.md) |

### Integration with inference engines

| Behaviour | Where | Explained in |
|---|---|---|
| The vLLM KV connector, scheduler hooks, cache groups | `integration/vllm/` — `lmcache_mp_connector.py`, `vllm_multi_process_adapter.py`, `kv_cache_groups.py` | [vLLM integration](features/05-vllm-integration.md) |
| SGLang | `integration/sglang/` | [vLLM integration](features/05-vllm-integration.md) (comparison) |
| TensorRT-LLM, Atom | `integration/tensorrt_llm/`, `integration/atom/` | — |
| Device/backend detection, non-CUDA support | `v1/platform/` | [MP architecture](features/01-mp-architecture.md) |

### Operator-facing surfaces

| Behaviour | Where | Explained in |
|---|---|---|
| Any `lmcache <command>` | `cli/commands/` — one module or package per command | [CLI](features/10-cli.md) |
| Metrics, events, tracing, log surface | `v1/mp_observability/` | [Observability](features/09-observability.md) |
| Direct Python use without vLLM | `sdk/` | [#4591 deep dive](issue-deep-dives/medium-4591-sdk-output-tokens.md) |
| API-key auth | `security/`, `v1/transport/auth.py` | [Transport layer](features/06-transport-layer.md) |

---

## 3.4 Naming conventions worth knowing

These recur everywhere and decode a lot of filenames:

- **`L1`** = memory this process owns (host DRAM, device DAX, GDS-backed). **`L2`** = any
  external tier reached through an adapter. The number is a tier, not a cache level in the
  CPU sense.
- **`*_l2_adapter.py`** — an implementation of `L2AdapterInterface`. Adding a backend means
  adding one of these plus a `factory.py` entry.
- **`*_controller.py` / `storage_controllers/`** — a background thread that operates on the
  L1 manager (prefetch, store, evict). The interface is in
  [`storage_controller.py`](https://github.com/LMCache/LMCache/blob/53bde913de1adae2a380fe0c2de8d2b9355b4702/lmcache/v1/distributed/storage_controller.py):
  *"Storage controllers are separate modules/threads that sees the L1 Manager and can
  operate on it."*
- **`modules/` under `multiprocess/`** — a server-side RPC handler group, registered into
  the module system. One file per RPC family.
- **`api.py` vs `internal_api.py`** — in `v1/distributed/`, `api.py` holds the types that
  cross module boundaries; `internal_api.py` holds the interfaces implementations must
  satisfy (`L1ManagerInterface`, and friends). If you are writing an adapter or manager,
  `internal_api.py` is your contract.
- **`base.py`** — the abstract base or shared state machine for a family (`l2_adapters/base.py`,
  `l1_manager/base.py`, `serde/base.py`, `cli/commands/base.py`).
- **`factory.py`** — the string-to-class mapping for a pluggable family. This is where a
  config value becomes an object, and the fastest way to enumerate what exists.
- **`_native` / `lmcache_native`** — the compiled extension. `lmcache_native.pyi` at the
  package root is the type stub; the implementation is in `csrc/lmcache_native/`.

---

## 3.5 Finding the design doc for a module

`docs/design/` **mirrors the `lmcache/` package tree**, which makes it mechanically
searchable. From [`docs/design/README.md`](https://github.com/LMCache/LMCache/blob/53bde913de1adae2a380fe0c2de8d2b9355b4702/docs/design/README.md):

> The design doc(s) for a given module live under the same relative path as the module's
> source. To find design docs for code at `lmcache/<path>/`, look under `docs/design/<path>/`.

So for code at `lmcache/v1/distributed/l2_adapters/`, read
`docs/design/v1/distributed/l2_adapters/`. A missing directory means no design doc exists,
not that you looked in the wrong place.

Two caveats, both from the README and both worth taking seriously:

> Not every module has a design doc — only modules whose design warrants prose beyond what
> the code and docstrings already communicate.

and module `README.md` files next to code stay in place and are symlinked *from*
`docs/design/<path>/README.md` — so some design-doc entries are views of a file that lives
with the code.

**Treat design docs as dated.** They describe intent at the time of writing; LMCache moves
fast. Where this guide found a design doc that no longer matches the code, it says so —
see [§9 of the L2 adapters document](features/04-l2-adapters-and-storage.md).

Highest-value design docs to read early:

| Doc | Why |
|---|---|
| `docs/design/v1/cache-identity.md` | How a cache entry is identified. Underpins everything. |
| `docs/design/v1/distributed/l2_adapters/overall.md` | The L2 contract, before the 25 implementations. |
| `docs/design/v1/distributed/l1_manager_interface.md` | The L1 contract. |
| `docs/design/integration/vllm/hybrid-kv-cache-groups.md` | Why KV cache groups exist; needed for hybrid models. |
| `docs/design/ARCHITECTURE_MULTI_HARDWARE.md` | How non-CUDA backends are abstracted. |

---

## 3.6 Where the tests are

`tests/` mirrors the package: `tests/v1/multiprocess/`, `tests/v1/distributed/`,
`tests/v1/storage_backend/`, `tests/sdk/`, `tests/cli/`. Two things to know before running
anything:

- **The default suite excludes the heavy directories.** From `AGENTS.md`, CI runs with
  `--ignore=tests/disagg --ignore=tests/v1/multiprocess/ --ignore=tests/v1/distributed/
  --ignore=tests/skipped --ignore=tests/v1/storage_backend/test_eic.py`. So the two largest
  engine directories are *not* in the default run; they need their own invocation and often
  real backends.
- **Multi-process and multi-node behaviour is tested in Buildkite**, not pytest — see
  `.buildkite/k3_tests/multiprocess`. If you are changing the MP server, the pytest suite
  is necessary and not sufficient.

---

## 3.7 Fastest ways to orient in unfamiliar code

Four moves that work well in this repo specifically:

1. **Read the `factory.py`.** For any pluggable family, the factory is the complete
   enumeration of what exists and the exact config strings that select it. Faster than
   grepping for classes.
2. **Read `internal_api.py` before any implementation.** The interface is short; the
   implementations are long and full of medium-specific detail that will mislead you about
   what is required.
3. **Check the mirrored design doc, then check its date.** `git log -1 --format=%ad` on the
   doc versus on the module tells you whether to trust it.
4. **Follow a request, not a class.** The useful spine is
   `integration/vllm/lmcache_mp_connector.py` → `vllm_multi_process_adapter.py` →
   `v1/client/` → `v1/transport/` → `v1/multiprocess/modules/lookup.py` →
   `v1/distributed/storage_manager.py` → `l1_manager/` / `l2_adapters/`. That chain is laid
   out in [the architecture document](02-architecture.md).

---

## 3.8 Where to look next

- [Architecture](02-architecture.md) — the process topology and the end-to-end request flow through the files above.
- [Project overview](01-project-overview.md) — the vocabulary (`chunk`, `realm`, `object group`, `L1`/`L2`) these paths assume.
- The feature documents linked throughout §3.3, one per subsystem.
