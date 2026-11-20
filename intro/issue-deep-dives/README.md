# Issue deep dives

Ten real, already-fixed LMCache bugs, each walked from symptom to merged diff.

Reading fixed bugs is the fastest way into an unfamiliar codebase. A feature document
tells you how a subsystem is *meant* to work; a bug tells you which of its invariants are
load-bearing, which assumptions were never written down, and what the code looked like to
the person who got it wrong. Every document here is built around a real merged commit, so
the claims are checkable.

See also the [PR walkthroughs](../pr-walkthroughs/README.md), which read three merged PRs as
design decisions rather than bug hunts.

---

## The ten

| Difficulty | Issue | Subsystem | What broke |
|---|---|---|---|
| **Difficult** | [#5151](difficult-5151-mamba-full-hit-restoration.md) | vLLM connector / lookup | A 100% cache hit restored a *mid-block* boundary, so Mamba models resumed from a recurrent state that was never snapshotted |
| **Difficult** | [#5713](difficult-5713-lora-cache-key-collision.md) | cache identity / LoRA | Every LoRA adapter shared the base model's cache keys, so a request could be served KV computed under different weights — silently |
| **Difficult** | [#6392](difficult-6392-prefetch-l1-landing-race.md) | prefetch / L1 | A chunk that arrived in L1 during the L2 lookup was reported as a miss and recomputed; one dropped chunk truncated the whole prefix |
| **Difficult** | [#5022](difficult-5022-turboquant-sm80-bfloat16.md) | TurboQuant serde | A Triton FP8 kernel could not *compile* on A100 for any bfloat16 model — and the failure was invisible |
| Medium | [#4416](medium-4416-percentile-off-by-one.md) | CLI / trace replay | A 1-based rank used as a 0-based index, so p90 and p99 reported the maximum |
| Medium | [#4591](medium-4591-sdk-output-tokens.md) | SDK | `output_tokens` returned the live sequence, so it included the prompt |
| Medium | [#4698](medium-4698-nixl-pool-slot-units.md) | NIXL L2 adapter | Two status fields the CLI subtracts were reported in different units, giving `-6 / 2 (-300%)` |
| Medium | [#5115](medium-5115-address-manager-zero-size.md) | memory allocator | A zero-byte allocation permanently fractured the free list — and the first fix turned that into a cache wipe |
| Medium | [#6543](medium-6543-zmq-grpc-fail-fast.md) | transport | A ZMQ client pointed at a gRPC server hung for 300 s, then reported the wrong cause |
| Medium | [#6698](medium-6698-blend-matcher-slot-ownership.md) | CacheBlend matcher | Evicting one copy of a chunk hid a different copy that was still stored — permanently |

### What "difficult" means here

Not "big diff" — the four difficult bugs have production diffs of **13**, **61**, **40** and
**3** lines. The difficulty is in the *diagnosis*:

- **#5151** needed a TP4 A100 box, a 27B hybrid model and a 100k-token prompt to see at all,
  and its most prominent clue (a chunk-count bracket) pointed at the wrong mechanism. Two
  well-reasoned hypotheses were eliminated by experiment before the real cause — a
  divisibility condition — was recognised.
- **#5713** produces *fluent, wrong output* with no error, needs multi-adapter traffic to
  occur at all, and the fix is inert without two other merged PRs.
- **#6392** is a race whose window spans an L2 lookup, invisible in any single-request test,
  and whose cost is amplified by prefix semantics.
- **#5022** failed three libraries deep, in Triton's code generator, on a hardware/dtype
  combination CI cannot reach, with a bare `AssertionError` and no message.

Most of the medium bugs are findable by one careful person reading one function — with the
caveat in #5115 §7 that its *label* said `good-first-issue` and its real fix needed deep
knowledge of error-type semantics across the whole allocation stack.

---

## Reading order

If you are working through these as a way to learn the codebase:

1. **[#4416](medium-4416-percentile-off-by-one.md)** — start here. Self-contained
   arithmetic, no LMCache concepts needed, and a test that asserted the wrong answer.
2. **[#4591](medium-4591-sdk-output-tokens.md)** — the SDK's token model, and the idea of a
   docstring as an unenforced contract. **Reproduced** with executed before/after output.
3. **[#5115](medium-5115-address-manager-zero-size.md)** — the allocator, and why an
   exception *type* is a protocol. Also **reproduced**.
4. **[#4698](medium-4698-nixl-pool-slot-units.md)** — the L2 adapter contract and unit
   discipline. Its margin finding (the repo's SLF lint rule is disabled repository-wide) is
   verified with the project's own pinned `ruff`.
5. **[#6543](medium-6543-zmq-grpc-fail-fast.md)** — transport, and diagnosability as a
   feature.
6. **[#6698](medium-6698-blend-matcher-slot-ownership.md)** — CacheBlend's index, and
   stale-ownership bugs. Three lines, and the best small example here.
7. **[#5022](difficult-5022-turboquant-sm80-bfloat16.md)** — needs the serde layer. Best
   lesson on testing hardware-specific code without the hardware.
8. **[#6392](difficult-6392-prefetch-l1-landing-race.md)** — needs L1's state machine and
   the prefetch path.
9. **[#5713](difficult-5713-lora-cache-key-collision.md)** — needs cache identity and the
   vLLM connector.
10. **[#5151](difficult-5151-mamba-full-hit-restoration.md)** — needs the connector, chunk
    hashing and KV cache groups. Read [vLLM
    integration](../features/05-vllm-integration.md) first if those are new.

---

## Themes that recur

Reading all ten together, the same few shapes keep appearing.

**Two values that should agree, computed in two places.** #4698 (files vs pages), #4591
(`decoded_tokens` vs `output_tokens`), #4416 (docstring vs code), #5713 (lookup used the
LoRA tag, the coalesced store did not), #6698 (slot occupancy vs registration record). In
each case the fix makes one value *derive from* the other rather than be recomputed
alongside it. That is the durable repair for this shape, and it is almost always smaller
than the alternative.

**The identity case hides the bug.** `pages_per_file` defaults to 1 (#4698); fractional
percentile ranks are unaffected by an off-by-one (#4416); non-chunk-multiple prompt lengths
never trigger the full-hit path (#5151); `size > 0` is the normal case (#5115); a single
LoRA adapter is self-consistent (#5713). In all of them the *common* configuration is
precisely the one where the defect cancels out — so the tests passed and production looked
fine.

**Correct error handling is not the same as a visible failure.** #5022's exception handler is
textbook — narrow `except`, `logger.exception` inside it — and the outcome was still a
silently degraded cache. #5115 goes further: the *first* fix handled the error correctly and
thereby caused a full cache eviction, because `RuntimeError` already meant "memory pressure"
to every backend.

**The most prominent clue can be an artifact.** #5151's "breaks between 114 and 128 chunks"
came from a transcription slip (an output recorded in an input column) and sustained two
wrong hypotheses for three days. The real signal was a one-token counter discrepancy sitting
in the same table, too small to be interesting.

**A metric derived from the buggy expression cannot detect the bug.** #5151's hit counter
came from `len(chunk_hashes)`, the same quantity the bug corrupted, so it reported a perfect
hit while restoring the wrong boundary. #6392's `l1_contended` counter *did* fire, but its
description called the event routine, so nobody read it as lost work.

**Good reports do the fixer's design work.** #4591 named the file, the line, the expected
value, *why the obvious fix is wrong*, and a list of regression tests — and the merged PR
implements that specification almost verbatim. #4698 named both files and forbade the
tempting wrong fix (clamping at the display layer). #5115 shipped a runnable repro and the
revision it was tested on. #6543 named the mechanism, the already-fixed reverse case as
precedent, *and* the pending default change that made it urgent — and merged the same day.

**Concurrency bugs are usually about intent, not locking.** #6392's `reserve_write` was
correctly locked throughout; it just answered "de-duplicate my write" when the caller meant
"get me a readable copy". #6698's slot clearing was inside the lock and still released a slot
it did not own.

---

## Honesty about verification

Each document states its own verification status in a block at the top. Summarising:

| Issue | Reproduced? | What was actually executed |
|---|---|---|
| #4591 | **Yes** | Real `sdk/request.py` at both revisions, run against the issue's repro + 3 more scenarios |
| #5115 | **Yes** | `AddressManager` extracted verbatim from both revisions and run; output matches the reporter's byte for byte |
| #4698 | No | Arithmetic re-derived by hand; **the SLF lint finding was verified** with the pinned `ruff` |
| #4416 | No | The two index expressions reimplemented and tabulated |
| #5151 | No | The two index expressions reimplemented; predicts the reporter's exact off-by-one counters |
| #5713 | No | Diff + the tag→key mechanism read at the pinned revision |
| #6392 | No | Diff + L1 state machine; the author's bench numbers are quoted as theirs |
| #6543 | No | Diff; HTTP/2 framing checked against RFC 9113 |
| #6698 | No | Diff + `matcher.py` data structures |
| #5022 | No | Nothing — no CUDA, no Triton in this environment |

No GPU, no multi-node cluster and no NIXL install were available, so no document claims a
performance number or an end-to-end reproduction it did not do. Where a conclusion rests on
reading code rather than running it, it says so; where an author measured something, it is
attributed to them.
