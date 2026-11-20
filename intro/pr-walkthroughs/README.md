# PR walkthroughs

Three merged pull requests read as **design decisions** rather than bug hunts.

The [issue deep dives](../issue-deep-dives/README.md) ask *"what broke, and why?"*. These ask
a different question: *"someone had to make a judgement call here — what were the options,
and what did they trade?"* The diffs are small; the reasoning around them is the content.

---

## The three

| PR | Area | The decision |
|---|---|---|
| [#6767](6767-blend-chunk-fingerprinting.md) | CacheBlend × Mamba hybrids | A conjunction (`all(object_groups)`) was written when every group committed every chunk. A group type arrived that commits one chunk by design. Who belongs in the quorum? |
| [#6754](6754-grpc-auto-server-mode.md) | gRPC transport | Changing a default from `sync` to "the best thing installed" — and what an `auto` value must *not* do to explicit ones |
| [#6752](6752-ci-stale-nightly-provenance.md) | CI / release engineering | A green nightly run that proved nothing for two months, because it recorded *when* it tested and never *what* it tested |

All three are merged, all three are permalinked, and #6767 is the commit this entire guide is
pinned to (`53bde913`).

---

## Why these three

They were not picked for size. The production diffs are **+32**, **+87** and **+308** lines
respectively, and the last one touches no `lmcache/` source at all. They were picked because
each turns on a decision that a reader will face again:

- **#6767 — scoping a quorum.** `all(...)` over a collection that can grow new *kinds* of
  member is a specification of who must agree, and it silently becomes wrong when a member
  arrives that abstains by design. The fix derives the set from the *consumer* instead of the
  producer.
- **#6754 — the semantics of "auto".** `auto` must resolve to the best available while
  explicit values stay strict, or there is no way to assert a requirement. Plus the thing
  nearly every default change forgets: the old default loses its test coverage the moment it
  stops being the default — a question the author raised in the PR rather than discovering
  later.
- **#6752 — what a passing test proves.** Validation has two timestamps (*when did we check*,
  *what did we check*) and most systems record only the first. This is the one PR here whose
  subject is the trustworthiness of the evidence everything else relies on.

---

## A theme they share with the deep dives

All three are instances of **degradation that is indistinguishable from health** — the
recurring shape in this codebase, noted in the
[deep-dive themes](../issue-deep-dives/README.md#themes-that-recur):

| PR | What looked fine | What was actually happening |
|---|---|---|
| #6767 | blend matches returning results | finding **one** chunk per request instead of all of them |
| #6754 | gRPC server serving requests | one client's queued work blocking every other client |
| #6752 | nightly CI green | testing a two-month-old wheel |

In a caching system almost every failure is a performance failure, and performance failures
do not raise. That is why so much of the reasoning in both folders is about **making the
invisible observable** — a metric description rewritten (#6392), provenance recorded
separately from validation time (#6752), a fallback logged at INFO (#6754).

---

## Verification status

Honest summary, since none of these could be run here:

| PR | Reproduced? | What was actually checked |
|---|---|---|
| #6767 | No — needs a Mamba hybrid + CacheBlend + H200 | Diff, and `_classify_cb_read_groups` at the pinned revision |
| #6754 | No — needs a running MP server and the optional `pygrpc-native` | Diff and config/help text |
| #6752 | No — needs the project's CI environment | Diff and PR description; the author's post-merge read-back is quoted as theirs |

No performance figure in these documents is mine. Where an author measured something it is
attributed to them and labelled.

---

## Where to look next

- [Issue deep dives](../issue-deep-dives/README.md) — ten already-fixed bugs, same treatment, bug-first.
- [Architecture](../02-architecture.md) and the [feature docs](../README.md#subsystem-deep-dives) — the subsystems these PRs touch.
