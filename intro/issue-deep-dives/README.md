# Issue deep dives

Five real, already-fixed LMCache bugs, each walked from symptom to merged diff.

Reading fixed bugs is the fastest way into an unfamiliar codebase. A feature document
tells you how a subsystem is *meant* to work; a bug tells you which of its invariants are
load-bearing, which assumptions were never written down, and what the code looked like to
the person who got it wrong. Every document here is built around a real merged commit, so
the claims are checkable.

---

## The five

| Difficulty | Issue | Subsystem | What broke |
|---|---|---|---|
| **Difficult** | [#5151](difficult-5151-mamba-full-hit-restoration.md) | vLLM connector / lookup | A 100% cache hit restored a *mid-block* boundary, so Mamba models resumed from a recurrent state that was never snapshotted |
| **Difficult** | [#5022](difficult-5022-turboquant-sm80-bfloat16.md) | TurboQuant serde | A Triton FP8 kernel could not *compile* on A100 for any bfloat16 model — and the failure was invisible |
| Medium | [#4416](medium-4416-percentile-off-by-one.md) | CLI / trace replay | A 1-based rank used as a 0-based index, so p90 and p99 reported the maximum |
| Medium | [#4591](medium-4591-sdk-output-tokens.md) | SDK | `output_tokens` returned the live sequence, so it included the prompt |
| Medium | [#4698](medium-4698-nixl-pool-slot-units.md) | NIXL L2 adapter | Two status fields the CLI subtracts were reported in different units, giving `-6 / 2 (-300%)` |

### What "difficult" means here

Not "big diff" — the two difficult bugs have production diffs of **13 lines** and **3
lines**. The difficulty is in the *diagnosis*:

- **#5151** needed a TP4 A100 box, a 27B hybrid model and a 100k-token prompt to see at
  all, and its most prominent clue (a chunk-count bracket) pointed at the wrong mechanism.
  Two well-reasoned hypotheses were eliminated by experiment before the real cause — a
  divisibility condition — was recognised.
- **#5022** failed three libraries deep, in Triton's code generator, on a hardware/dtype
  combination that CI cannot reach, with a bare `AssertionError` and no message.

The three medium bugs are each findable by one careful person reading one function.

---

## Reading order

If you are working through these as a way to learn the codebase:

1. **[#4416](medium-4416-percentile-off-by-one.md)** — start here. Self-contained
   arithmetic, no LMCache concepts needed, and a test that asserted the wrong answer.
2. **[#4591](medium-4591-sdk-output-tokens.md)** — introduces the SDK's token model and
   the idea of a docstring as an unenforced contract. The only deep dive here with
   **executed** before/after evidence.
3. **[#4698](medium-4698-nixl-pool-slot-units.md)** — introduces the L2 adapter contract
   and unit discipline. Its margin finding (the repo's SLF lint rule is disabled
   repository-wide) is verified with the project's own pinned `ruff`.
4. **[#5022](difficult-5022-turboquant-sm80-bfloat16.md)** — needs the serde layer. Best
   lesson on testing hardware-specific code without the hardware.
5. **[#5151](difficult-5151-mamba-full-hit-restoration.md)** — needs the vLLM connector,
   chunk hashing and KV cache groups. Read [vLLM
   integration](../features/05-vllm-integration.md) first if those are new.

---

## Themes that recur

Reading all five together, the same few shapes keep appearing:

**Two values that should agree, computed in two places.** #4698 (files vs pages), #4591
(`decoded_tokens` vs `output_tokens`), #4416 (docstring vs code). In each case the fix
makes one value *derive from* the other rather than be recomputed alongside it. That is
the durable repair for this shape, and it is almost always smaller than the alternative.

**The identity case hides the bug.** `pages_per_file` defaults to 1 (#4698); fractional
percentile ranks are unaffected by an off-by-one (#4416); non-chunk-multiple prompt lengths
never trigger the full-hit path (#5151). In all three, the common configuration is
precisely the one where the defect cancels out — so the tests passed and production looked
fine.

**Correct error handling is not the same as a visible failure.** #5022's exception handler
is textbook — narrow `except`, `logger.exception` inside it — and the outcome was still a
silently degraded cache, because in a tiered system a dead tier is indistinguishable from an
unconfigured one.

**The most prominent clue can be an artifact.** #5151's "breaks between 114 and 128 chunks"
came from a transcription slip (an output recorded in an input column) and sustained two
wrong hypotheses for three days. The real signal was a one-token counter discrepancy sitting
in the same table, too small to be interesting.

**Good reports do the fixer's design work.** #4591 named the file, the line, the expected
value, *why the obvious fix is wrong*, and a list of regression tests — and the merged PR
implements that specification almost verbatim. #4698 named both files and forbade the
tempting wrong fix (clamping at the display layer). Neither needed the maintainer to
reproduce anything.

---

## Honesty about verification

Each document states its own verification status in a block at the top. Summarising:

| Issue | Reproduced? | What was actually executed |
|---|---|---|
| #4591 | **Yes** | Real `request.py` at both revisions, run against the issue's repro + 3 more scenarios |
| #4698 | No | Arithmetic re-derived by hand; **the SLF lint finding was verified** with the pinned `ruff` |
| #4416 | No | The two index expressions reimplemented and tabulated |
| #5151 | No | The two index expressions reimplemented; predicts the reporter's exact off-by-one counters |
| #5022 | No | Nothing — no CUDA, no Triton in this environment |

No GPU, no multi-node cluster and no NIXL install were available, so no document claims a
performance number or an end-to-end reproduction it did not do. Where a conclusion rests on
reading code rather than running it, it says so.
