# Issue #4546 — Disk admission capacity is never released on removal

- **Issue:** [#4546](https://github.com/LMCache/LMCache/issues/4546)
- **Branch:** `fix/4546-disk-size-accounting`
- **Base:** `dev` @ `ee68e5ba`
- **Files touched:** `lmcache/v1/storage_backend/local_disk_backend.py`, `tests/v1/storage_backend/test_local_disk_put_admission.py`
- **Severity:** permanent, silent loss of the disk cache tier

---

## 1. Problem

`LocalDiskBackend` tracks consumed disk bytes in `current_cache_size`, and that
counter gates every write in `submit_put_task`:

```python
required_eviction = self.current_cache_size + required_size - self.max_cache_size
if required_eviction > 0:
    ...
while self.current_cache_size + required_size > self.max_cache_size:
    evict_keys = self.cache_policy.get_evict_candidates(self.dict, num_candidates=1)
    if not evict_keys:
        logger.warning("No eviction candidates found. Disk space under pressure.")
        return None
    self.batched_remove(evict_keys, force=False)

self.current_cache_size += required_size
```

It was incremented on every put, but decremented in **only one place**: the
internal eviction loop. `remove()` — which defaults to `force=True` and is the
path every *external* caller uses (explicit invalidation, lookup-miss cleanup)
— decremented a *different* counter and left the gate untouched:

```python
self.usage -= size
self.stats_monitor.update_local_storage_usage(self.usage)
# current_cache_size: never touched
os.remove(path)
```

## 2. Consequence

Every external removal leaks phantom bytes. `current_cache_size` only ever
grows, regardless of what is actually on disk.

Once the accumulated drift reaches `max_cache_size`, the admission check can
never pass again:

1. `required_eviction > 0` is permanently true.
2. The eviction loop asks for candidates, but the dict is genuinely (or nearly)
   empty — the entries were already removed.
3. `get_evict_candidates` returns nothing → `return None`.

So **every subsequent disk put is silently dropped**, while the log insists
*"No eviction candidates found. Disk space under pressure."* — on a disk that
may hold nothing at all.

Three properties make this unusually bad:

- **Silent.** `submit_put_task` returns `None`; no exception reaches the caller.
- **Permanent.** Nothing ever decrements the drift back down. Only a restart
  clears it.
- **Actively misleading.** The log blames disk pressure, pointing any
  investigation at capacity planning rather than at an accounting bug.

## 3. The trap: double-decrementing

The naive fix — decrement inside `remove()` — **double-counts on the eviction
path**, because the eviction loop already deducted the size itself before
delegating:

```python
for evict_key in evict_keys:
    self.current_cache_size -= self.dict[evict_key].size   # deduction #1
self.batched_remove(evict_keys, force=False)               # would be #2
```

Over-decrementing is arguably worse than the original bug: the backend would
believe it has free space it does not have, and the disk would grow past
`max_cache_size` without bound.

### Tracing every deletion path

- `batched_remove` is **not** overridden here. The base implementation in
  `abstract_backend.py` is simply a loop:

  ```python
  for key in keys:
      num_removed += self.remove(key, force=force)
  ```

- `force` does **not** affect accounting at all. It selects the lock, because
  the eviction loop already holds `disk_lock`:

  ```python
  lock_context = self.disk_lock if force else nullcontext()
  ```

So `remove()` is the **single funnel** for both external invalidation and
internal eviction. That makes it the correct and only home for the deduction.

## 4. Implementation

Decrement in `remove()`, beside the existing `usage` decrement and inside the
same lock context:

```python
self.usage -= size
self.stats_monitor.update_local_storage_usage(self.usage)

# Sole owner of this deduction: every deletion path funnels through
# here, so a caller deducting `size` too would free the same bytes
# twice and let the disk grow past max_cache_size.
self.current_cache_size -= size
```

…and delete the eviction loop's own deduction:

```diff
-                    for evict_key in evict_keys:
-                        self.current_cache_size -= self.dict[evict_key].size
-
                     self.batched_remove(evict_keys, force=False)
```

The comment is deliberate and explains **why**, not what: the double-decrement
trap is non-obvious, and a future editor "restoring" the loop's deduction would
silently reintroduce unbounded disk growth. This is exactly the case the repo's
standard reserves comments for.

## 5. Known consequence worth a reviewer's eye

The eviction loop's **termination condition** now depends on `remove()`
performing the deduction:

```python
while self.current_cache_size + required_size > self.max_cache_size:
```

Previously the loop's own unconditional deduction guaranteed progress. Now, if
`remove()` returned `False` without deducting, the loop would not converge.

Why it cannot happen in practice: candidates come from
`get_evict_candidates(self.dict, ...)`, called with that same dict under the
same `disk_lock`, so a candidate is always resident and `remove()` always
deducts. Per the repo's standard ("no handling for failures that cannot
occur"), no guard was added — but the coupling is tighter than before, so it is
called out explicitly in the PR rather than left for a reviewer to find.
`test_eviction_does_not_release_the_same_bytes_twice` exercises the loop end to
end, so a convergence failure would surface as a hanging test rather than
silently.

## 6. Two things deliberately NOT changed

### `usage` vs `current_cache_size` — not redundant

They look interchangeable and are not. They are incremented at **different
lifecycle points**:

| Counter | Incremented | Meaning |
|---|---|---|
| `current_cache_size` | `submit_put_task`, at admission | bytes **reserved**, including in-flight writes |
| `usage` | `async_save_bytes_to_disk`, after the write | bytes **actually written** |

A reservation counter and a realised-usage counter legitimately differ while
writes are in flight. Unifying them is a separate refactor with its own
correctness argument; bundling it would have put the bugfix at risk.

### An adjacent leak in the read path

`async_load_bytes_from_disk` pops a dict entry directly when the file has
vanished from disk underneath it:

```python
except FileNotFoundError:
    logger.warning("File not found on disk: %s", path)
    if self.dict.get(key, None):
        self.dict.pop(key)
```

This leaks **both** counters. It is a real, pre-existing bug found while
auditing every dict mutation for this fix — but it is a *different* defect, and
it cannot simply route through `remove()` because `remove()` calls `os.remove`
on the very file that is already missing. Left alone to keep this PR to one
defect, and disclosed in the PR description rather than quietly ignored.

## 7. Tests

Appended to `tests/v1/storage_backend/test_local_disk_put_admission.py`, whose
existing scaffolding already models the admission gate. Its `_create_backend`
helper stubs `batched_remove`, which would hide the exact interaction under
test, so a sibling helper rebuilds the backend with the **real**
`remove()`/`batched_remove()` plus the fields they need (`usage`,
`stats_monitor`, `batched_msg_sender`) and real files on disk for `os.remove`.

| Test | Asserts | On `dev` |
|---|---|---|
| `test_external_remove_releases_tracked_bytes` | `current_cache_size` returns to its pre-put value | **fails** (stays 100) |
| `test_put_is_admitted_again_after_external_remove` | the admission gate reopens | **fails** (put dropped) |
| `test_eviction_does_not_release_the_same_bytes_twice` | each freed chunk accounted once; loop converges | passes (regression guard) |

The third test passes on `dev` by design — it guards the double-decrement trap
introduced by the fix, so it must hold both before and after.

## 8. Verification

```console
$ pytest -q tests/v1/storage_backend/test_local_disk_put_admission.py \
          tests/v1/storage_backend/test_local_disk_backend.py
43 passed

$ ruff check lmcache/v1/storage_backend/local_disk_backend.py \
             tests/v1/storage_backend/test_local_disk_put_admission.py
All checks passed!

$ ruff format --check <same files>
2 files already formatted

$ codespell --toml pyproject.toml <changed files>
(clean)
```

**Regression tests confirmed meaningful** — restoring the original source from
`dev` and re-running:

```
2 failed, 6 passed
```

Exactly the two accounting tests fail; the double-decrement guard passes either
way, as intended.

`mypy` reports one pre-existing error in this file (`insert_key` receiving
`dtype | None`, line 714 on `dev` → 716 here, shifted only by added lines).
Confirmed present on untouched `dev`; unrelated to this change.

---

## 9. PR description

> **What this PR does / why we need it**:
>
> Fixes #4546.
>
> `LocalDiskBackend.current_cache_size` gates every disk put, but only the
> internal eviction loop ever deducted from it. `remove()` decremented the
> `usage` gauge and left the gate untouched — and `remove()` (default
> `force=True`) is the path every *external* caller uses.
>
> So the counter only ever grew. Once the accumulated drift reached
> `max_cache_size`, the admission check in `submit_put_task` could never pass
> again: the backend silently dropped **every** subsequent put while logging
> "No eviction candidates found. Disk space under pressure." — on a disk that may
> hold nothing at all. The disk tier stays dead until the process restarts, and
> nothing surfaces the cause.
>
> **Special notes for your reviewers**:
>
> The trap here is double-counting. The eviction loop deducted each chunk's size
> itself *before* delegating to `batched_remove(..., force=False)`, and
> `batched_remove` just loops over `remove()`. `force` only selects the lock
> (`disk_lock` vs `nullcontext`), not the accounting, so `remove()` is the single
> funnel for both external and eviction deletes. It now owns the deduction and
> the loop's own deduction is gone; keeping both would free the same bytes twice
> and let the disk grow past its limit.
>
> One consequence worth your eye: the eviction loop's termination condition now
> depends on `remove()` performing the deduction. Candidates come from
> `get_evict_candidates(self.dict)` under the same lock, so a candidate is always
> resident and `remove()` always deducts — but it is a tighter coupling than
> before and I would rather you confirm you are happy with it.
>
> `usage` is deliberately left alone. It looks redundant with
> `current_cache_size`, but they are incremented at different points —
> `current_cache_size` reserves space at admission time, `usage` counts bytes
> actually written — so they are not interchangeable. Unifying them is a separate
> refactor.
>
> **Adjacent leak, not fixed here:** the `FileNotFoundError` recovery in
> `async_load_bytes_from_disk` pops the dict entry directly, leaking *both*
> counters. It cannot route through `remove()` because `remove()` itself calls
> `os.remove` on the already-missing file. Left alone to keep this PR to one
> defect — happy to file it separately.
>
> ```
> $ pytest -q tests/v1/storage_backend/test_local_disk_put_admission.py \
>          tests/v1/storage_backend/test_local_disk_backend.py
> 43 passed
> ```
>
> The two accounting tests fail against `dev`; the third is a regression guard
> for the double-deduction and passes either way, which is the point.
>
> **If applicable**:
>
> - [ ] this PR contains user facing changes - docs added
> - [x] this PR contains unit tests
