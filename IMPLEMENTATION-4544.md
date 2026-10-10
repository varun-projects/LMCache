# Issue #4544 — Generator guard falls through in `_fast_split_by_subtensor`

- **Issue:** [#4544](https://github.com/LMCache/LMCache/issues/4544)
- **Branch:** `fix/4544-token-database-guard-return`
- **Base:** `dev` @ `ee68e5ba`
- **Files touched:** `lmcache/v1/token_database.py`, `tests/v1/test_token_database.py`
- **Severity:** silent cache corruption (one path), crash (the other)

---

## 1. Problem

`SegmentTokenDatabase._fast_split_by_subtensor` splits a token tensor on a
separator subsequence. It opens with a guard for degenerate input:

```python
def _fast_split_by_subtensor(self, tokens):
    if self.sep_len == 0 or len(tokens) < self.sep_len:
        yield tokens
        # <-- no return

    windows = tokens.unfold(0, self.sep_len, 1)
    matches = (windows == self.sep_tokens).all(dim=1).nonzero(as_tuple=True)[0].tolist()
    ...
```

The function is a **generator**. `yield` *suspends* a generator; it does not
return from it. So the guard yields `tokens` and then, the moment the consumer
asks for a second chunk, execution **resumes into the very code the guard
exists to prevent**.

In a non-generator function `yield`'s absence of `return` would be a visible
bug. Here it reads like an early exit and behaves like a fall-through — which
is exactly why it survived review.

## 2. The two failure modes

### 2a. `len(tokens) < sep_len` — loud

`unfold(0, size, 1)` requires `size <= tokens.size(0)`:

```
RuntimeError: maximum size for tensor at dimension 0 is 2 but size is 4
```

Recoverable in the sense that it is impossible to miss.

### 2b. `sep_len == 0` — silent, and the dangerous one

`unfold(0, 0, 1)` does **not** raise. It produces a tensor with a zero-width
second dimension — shape `(len(tokens) + 1, 0)`. Then:

```python
(windows == self.sep_tokens).all(dim=1)
```

`.all()` over an **empty** dimension is **vacuously true**. So every window
"matches" the separator, at every position. The split loop then emits a chunk
boundary at every index, turning one request into `len(tokens) + 1` chunks —
each of which is hashed and stored as if it were a genuine semantic segment.

No exception. No warning. Just a cache full of garbage keys and a blend feature
that silently does nothing resembling its contract.

### Reachability of `sep_len == 0`

Not hypothetical. The separator is built in the constructor:

```python
# TODO (Jiayi): figure out how to decide when
# to use `1:` (whether there's a special starting token
# in the beginning)
self.sep_tokens = self.tokenizer.encode(config.blend_special_str)[1:]
```

The `[1:]` strips an assumed leading BOS-style token. When
`blend_special_str` encodes to a **single** id, that strip leaves an empty
list — `sep_len == 0`. The `TODO` directly above acknowledges the heuristic is
unresolved, so this is a live configuration, not a theoretical one.

## 3. Implementation

### 3a. Exit the guard

```python
if self.sep_len == 0 or len(tokens) < self.sep_len:
    yield tokens
    return
```

A bare `return` was chosen over restructuring (e.g. `if/else`) because
yielding the whole tensor unsplit is plainly the behaviour the guard already
intended. The fix is to stop falling through, not to redefine the contract.

The docstring now states that contract explicitly, so the next reader does not
have to infer it:

```
When there is no usable separator, or `tokens` is shorter than the
separator, `tokens` is yielded unsplit as the only chunk.
```

### 3b. Surface the misconfiguration

```python
if self.sep_len == 0:
    logger.warning(
        "blend_special_str %r encodes to a single token, which the "
        "leading-token strip removes. No separator is left, so every "
        "request will be cached as one segment instead of being split.",
        config.blend_special_str,
    )
```

**Why this is included.** Fixing only the fall-through turns a loud corruption
into a *silent* no-op: blending is configured, appears to work, and quietly
never splits anything. The operator has no way to see it. One warning at
construction (not per request) converts an invisible failure into a diagnosable
one.

**Why it might be dropped.** It is arguably a second concern. The PR
description says so explicitly and offers to remove it — better to be told
"out of scope" than to have smuggled it in.

### 3c. Consumer verified, not touched

`process_tokens` offsets chunk boundaries by the separator width, but only
after the first chunk:

```python
if idx > 0:
    start_idx += self.sep_len
    end_idx += self.sep_len
```

With the guard now exiting, both degenerate paths produce **exactly one**
chunk, so `idx > 0` never fires and the offsets are correct by construction.
Checked deliberately — a fix that corrected the split but left the consumer
mis-indexing would have been worse than the original bug.

## 4. Tests

Appended to the end of `tests/v1/test_token_database.py`.

| Test | Covers | On `dev` |
|---|---|---|
| `test_segment_db_keeps_input_shorter_than_separator_whole` | mode 2a | **fails** (`RuntimeError`) |
| `test_segment_db_keeps_input_whole_when_separator_is_empty` | mode 2b | **fails** (6 chunks, not 1) |
| `test_segment_db_splits_on_separator` | normal path regression guard | passes (by design) |

Two deliberate choices:

- **No Hugging Face dependency.** The existing `test_segment_token_database` is
  gated behind `@pytest.mark.skipif(not hf_credentials_available())` and
  therefore skips in most environments — including CI without credentials. A
  stub tokenizer (`_FixedIdsTokenizer`) is injected via `monkeypatch`, so these
  tests always run.
- **Public surface only.** Assertions go through `process_tokens`, the public
  entry point, rather than calling the private generator directly. The
  `len(tokens) + 1` shredding is observable there as a result count, which is
  the contract a caller actually depends on.

### Naming, to avoid a known conflict

Open PR **#5584** also edits this file and adds a helper named `_FakeTokenizer`
immediately after `test_segment_token_database`. To minimise merge pain, these
tests are appended **at the very end** of the file and the helper is named
**`_FixedIdsTokenizer`**.

### One test written and then removed

A `caplog`-based assertion on the 3b warning was dropped. LMCache's logger does
not propagate to pytest's capture fixture, so making it pass required
test-harness plumbing disproportionate to asserting a log line — and the repo's
own guidance is to test the documented behavioural contract, which a warning
string is not.

## 5. Verification

```console
$ pytest -q tests/v1/test_token_database.py
10 passed, 8 skipped

$ ruff check lmcache/v1/token_database.py tests/v1/test_token_database.py
All checks passed!

$ ruff format --check lmcache/v1/token_database.py tests/v1/test_token_database.py
1 file already formatted

$ mypy --config-file=pyproject.toml lmcache/v1/token_database.py tests/v1/test_token_database.py
Success: no issues found in 2 source files

$ codespell --toml pyproject.toml <changed files>
(clean)
```

The 8 skips are the pre-existing HF-credential-gated test, unchanged.

**Regression tests confirmed meaningful** — reverting only the `return` and
re-running:

```
2 failed, 8 passed, 8 skipped
```

Both degenerate-path tests fail, and the normal-path test still passes. A test
that cannot fail is not a regression test, so this check was run rather than
assumed.

---

## 6. PR description

> **What this PR does / why we need it**:
>
> Fixes #4544.
>
> `SegmentTokenDatabase._fast_split_by_subtensor` is a generator, so the
> `yield tokens` in its degenerate-input guard only suspended execution. As soon
> as the consumer asked for the next chunk, control resumed into the
> sliding-window match the guard existed to prevent.
>
> Two failure modes follow:
>
> - `len(tokens) < sep_len` — `tokens.unfold(0, sep_len, 1)` raises
>   `RuntimeError: maximum size for tensor at dimension 0 is N but size is M`.
> - `sep_len == 0` — `unfold(0, 0, 1)` succeeds and produces a zero-width
>   dimension, so `.all(dim=1)` is **vacuously true** at every position. Every
>   index reports a separator match and a single request is split into
>   `len(tokens) + 1` chunks, each hashed and cached as though it were a real
>   segment. This one is silent: no exception, just corrupted segmentation.
>
> **Special notes for your reviewers**:
>
> `sep_len == 0` is reachable, not hypothetical: `sep_tokens` is built as
> `tokenizer.encode(config.blend_special_str)[1:]`, and the `[1:]` leaves nothing
> when the string encodes to a single id (the `TODO (Jiayi)` above that line
> acknowledges the heuristic).
>
> Two judgment calls worth a second opinion:
>
> 1. **`return` rather than restructuring.** Yielding the whole tensor unsplit is
>    the behaviour the guard clearly intended, so the fix is to stop falling
>    through rather than to change the contract.
> 2. **A warning at construction for `sep_len == 0`.** Silently treating every
>    request as one segment hides a config error the operator cannot otherwise
>    see — blending is simply disabled. I judged a one-time warning in scope
>    because it is the same defect; happy to drop it if you would rather keep this
>    PR purely about the fall-through.
>
> The consumer in `process_tokens` offsets by `sep_len` only when `idx > 0`. With
> the guard exiting correctly both degenerate paths produce exactly one chunk, so
> that branch never fires and the offsets stay correct.
>
> New tests use a stub tokenizer, so unlike `test_segment_token_database` they do
> not need Hugging Face credentials. Both degenerate-path tests fail on `dev` and
> pass here; the splitting test guards the normal path and passes either way.
>
> ```
> $ pytest -q tests/v1/test_token_database.py
> 10 passed, 8 skipped          # 8 skipped = existing HF-gated test, no credentials
> ```
>
> **If applicable**:
>
> - [ ] this PR contains user facing changes - docs added
> - [x] this PR contains unit tests
