# Issue #4506 — Lazy `%`-format logging in the Mooncake store connector

- **Issue:** [#4506](https://github.com/LMCache/LMCache/issues/4506) (`good first issue`, umbrella [#3372](https://github.com/LMCache/LMCache/issues/3372))
- **Branch:** `gfi/4506-mooncakestore-lazy-logging`
- **Base:** `dev` @ `ee68e5ba`
- **Files touched:** `lmcache/v1/storage_backend/connector/mooncakestore_connector.py`, `pyproject.toml`
- **Behaviour change:** none

---

## 1. Problem

`mooncakestore_connector.py` emitted every log record through an f-string:

```python
logger.debug(f"Using batch_get_into for {len(keys)} keys (zero-copy mode)")
```

An f-string is evaluated **before** `logger.debug` is called, so the string is
built unconditionally — even when the record is immediately discarded because
the logger's level is above `DEBUG`. The `%`-format form defers interpolation
into the logging layer, which only formats if a handler will actually emit:

```python
logger.debug("Using batch_get_into for %d keys (zero-copy mode)", len(keys))
```

This matters more here than in a typical module because several of the
converted calls sit on the **batched get path**, which runs per request.

The repository already enforces this with ruff rule `G004`
(`logging-f-string`), but this file carried a blanket exemption in
`pyproject.toml`, so the rule never checked it.

## 2. Root cause

Not a logic bug — accumulated drift. The file was added (and grown) while its
`G004` entry in `[tool.ruff.lint.per-file-ignores]` suppressed the lint, so new
f-string log calls kept landing without resistance.

## 3. Implementation

Two coordinated changes:

1. **Converted all 23 f-string log calls** to `%`-format.
2. **Deleted the file's `G004` per-file-ignore** so the rule now guards it.

Step 2 is what makes this a permanent fix rather than a one-time cleanup — it
is the burn-down pattern the umbrella issue describes, and without it the file
would silently regress.

### Conventions followed

Taken from the already-merged precedent PR #5094 (same cleanup, different
file), so this change stays consistent with the rest of the series:

| Case | Conversion |
|---|---|
| Most values, including ints | `%s` |
| Explicit `:.2f` in the f-string | `%.2f` |
| `f"...{str(exc)}"` | `"...%s", exc` — drop the redundant `str()`, the logging layer applies it |
| Counts where `%d` reads clearer | `%d` |

Multi-line calls keep their implicit string concatenation and move the
arguments after the format string:

```python
logger.error(
    "Metadata required for batch_get_into but not available: "
    "meta_shapes=%s, "
    "meta_dtypes=%s, "
    "meta_fmt=%s",
    self.meta_shapes,
    self.meta_dtypes,
    self.meta_fmt,
)
```

### Counting note

A naive single-line grep (`logger\.[a-z]*(\s*f"`) finds only **11** call sites
and suggests the issue's count of 23 is stale. It is not: **12 of the 23 calls
are multi-line**, with the f-string starting on the line after `logger.x(`. The
issue's number was correct. `ruff --select G004` is the reliable oracle here,
not grep.

## 4. Deliberately out of scope

- **`logger.error` → `logger.exception` inside `except` blocks.** The repo's
  coding standard prefers `logger.exception` there, and several converted calls
  are in `except` blocks. But `logger.exception` appends a traceback, which
  **changes output** — that does not belong in a change whose entire claim is
  "rendered messages are identical". Separate concern, separate PR.
- **Non-logging f-strings.** Untouched; `G004` only concerns logging calls.
- **Other files' `G004` exemptions.** Still present, one file per PR per the
  umbrella issue's burn-down approach.

## 5. Verification

```console
$ ruff check --isolated --select G004,PLE1205 \
    lmcache/v1/storage_backend/connector/mooncakestore_connector.py
All checks passed!

$ ruff check lmcache/v1/storage_backend/connector/mooncakestore_connector.py
All checks passed!

$ ruff format --check lmcache/v1/storage_backend/connector/mooncakestore_connector.py
1 file already formatted

$ codespell --toml pyproject.toml <changed files>
(clean)
```

`PLE1205` (`logging-too-many-args`) is included deliberately: it is the check
that catches the classic mistake in this conversion — a placeholder count that
no longer matches the argument count.

Repo-wide `ruff check .` reports **11 errors**, which is **identical to
untouched `dev`**. All 11 pre-exist and none are in the touched files.

`mypy` reports one pre-existing error in this file
(`RemoteMetadata.deserialize` receiving `memoryview[int]`, line 654 on `dev`
→ 668 here, shifted only by added lines). Confirmed present on untouched `dev`;
unrelated to this change.

### No tests

Intentional, matching every merged PR in this series (#5094, #5014, #4785,
#5223, #5421). The change is a mechanical no-op on rendered output, and `G004`
plus `PLE1205` in CI are the regression guard. A unit test asserting log text
would test the logging library, not this code.

### Not claimed

No benchmark was run. This PR does **not** claim a measured throughput or
latency improvement — only that the formatting work is no longer unconditional.
Mirrors the same bounded claim made in #5094.

---

## 6. PR description

> **What this PR does / why we need it**:
>
> Refs #4506. Refs #3372.
>
> `mooncakestore_connector.py` built every log message eagerly with f-strings, so
> the interpolation ran even when the record was discarded by level. Several of
> these sit on the batched get path at `debug` level, where the formatting cost
> was paid on every request regardless of the configured level.
>
> All 23 call sites now use `%`-format so the logging layer defers interpolation,
> and the file's `G004` entry in `[tool.ruff.lint.per-file-ignores]` is removed so
> the rule keeps it that way.
>
> **Special notes for your reviewers**:
>
> Rendered messages are byte-for-byte unchanged; this is a mechanical conversion
> with no behaviour change. The redundant `str()` around exception objects is
> dropped, matching #5094, since the logging layer applies `str()` itself.
>
> `logger.error` inside `except` blocks was deliberately left as-is rather than
> converted to `logger.exception` — that changes output by adding a traceback, so
> it belongs in its own change, not a lazy-logging cleanup.
>
> Verified:
>
> ```
> $ ruff check --isolated --select G004,PLE1205 lmcache/v1/storage_backend/connector/mooncakestore_connector.py
> All checks passed!
> $ ruff check . && ruff format --check .
> # 11 pre-existing errors, identical to untouched dev (none in the touched files)
> ```
>
> No benchmark was run, so this PR does not claim a measured throughput or
> latency improvement — only that the formatting work is no longer unconditional.
>
> **If applicable**:
>
> - [ ] this PR contains user facing changes - docs added
> - [ ] this PR contains unit tests
