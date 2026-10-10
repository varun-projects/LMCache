# Issue #4591 — `output_tokens` returns the prompt as well as the generation

- **Issue:** [#4591](https://github.com/LMCache/LMCache/issues/4591)
- **Branch:** `fix/4591-sdk-output-tokens`
- **Base:** `dev` @ `ee68e5ba`
- **Files touched:** `lmcache/sdk/request.py`, `tests/sdk/test_request.py` (new), `docs/design/sdk/request.md`
- **Public API behaviour change:** yes

---

## 1. Problem

`LMCacheRequestStream` exposes three accessors describing what a request
produced:

```python
@property
def decoded_tokens(self) -> int:
    """Return cumulative tokens decoded across all segments so far."""
    return self._decoded

@property
def output_text(self) -> str:
    """Return the concatenated generated text across all segments."""
    return "".join(self._text_parts)

@property
def output_tokens(self) -> list[int]:
    """Return the concatenated generated tokens across all segments."""
    return self.tokens        # <-- the live sequence: prompt + generated
```

`self.tokens` is the sequence backing the stored KV. It starts as the prompt
and is extended with each generated token. So `output_tokens` returned the
**prompt concatenated with the generation**, directly contradicting its own
docstring.

Two concrete inconsistencies a caller hits immediately:

- `len(stream.output_tokens) != stream.decoded_tokens`
- `stream.output_tokens` and `stream.output_text` describe **different spans** —
  pairing them (the obvious thing to do) silently misaligns text with ids.

For a 3-token prompt and 2 generated tokens, `output_tokens` returned 5 ids
while `output_text` covered 2 and `decoded_tokens` reported 2.

## 2. Root cause

`output_text` and `decoded_tokens` each have a dedicated accumulator
(`_text_parts`, `_decoded`) appended to inside `generate()`. `output_tokens`
never got one — it reached for the nearest already-existing list that happened
to contain generated ids, which also happened to contain the prompt.

## 3. Why slicing is the wrong fix

The tempting one-liner is:

```python
return self.tokens[self._prompt_len:]     # WRONG
```

It does not work, because `self.tokens` is **not append-only**. `update()`
replaces it wholesale:

```python
def update(self, kind, kv, tokens) -> None:
    ...
    self.tokens = list(tokens)
    self.done = False
```

`update()` is how an edited KV is stored, and `modify_kv()` calls it after
handing the cached prefix to a caller-supplied editing function (compaction,
token dropping — see `examples/token_dropping/`). After such an edit:

- the sequence may be **shorter** than the prompt,
- it may contain **entirely different ids**,
- the prompt prefix is **not recoverable** from it at all.

Any prompt-length arithmetic is therefore wrong by construction the moment the
KV is modified — which is the SDK's headline feature, not an edge case.

## 4. Implementation

Mirror the pattern the sibling accessors already use: a dedicated accumulator.

```python
# __init__
self._text_parts: list[str] = []
self._token_parts: list[int] = []          # added
```

```python
# generate(), inside the existing finally:
finally:
    self._decoded += len(gen_tokens)
    self._text_parts.extend(gen_texts)
    self._token_parts.extend(gen_tokens)   # added
    self.tokens.extend(gen_tokens)
```

```python
@property
def output_tokens(self) -> list[int]:
    """Return the concatenated generated tokens across all segments."""
    return list(self._token_parts)
```

Three details that are deliberate:

1. **Extended inside the existing `finally`**, alongside the other two
   accumulators. If the engine raises partway through a stream, the tokens
   already decoded stay recorded — consistent with `_decoded` and
   `_text_parts`, which behave that way today. Putting it in the `try` would
   have made the three accessors disagree on failure.
2. **Returns a copy** (`list(...)`), not the internal list. `output_text`
   returns a fresh `str` and `decoded_tokens` an `int`, both immutable; handing
   out a live list would let a caller corrupt the history by accident.
3. **`self.tokens` is untouched.** It remains the live KV-backing sequence.
   Anyone who genuinely wants prompt + generation reads `.tokens`, which is
   unchanged — so the migration path is a one-word edit.

### Documentation

`docs/design/sdk/request.md` documents the state model, and its field list was
now incomplete. Updated to name the new accumulator **and record why it exists
separately**, which is the part a future reader would otherwise "simplify" away:

```markdown
- **`_decoded` / `_text_parts` / `_token_parts`** — cumulative generated count /
  text / token ids across all `generate`s (`decoded_tokens`, `output_text`,
  `output_tokens`); unaffected by compaction. Kept separate from `tokens`
  because `update` replaces the live sequence, which would otherwise lose or
  misattribute the generation history.
```

(`docs/design/` is Markdown, not Sphinx-built — only `docs/source/` requires a
Sphinx build, so no doc build was needed here.)

## 5. Tests

New file `tests/sdk/test_request.py`, matching the style of the neighbouring
`test_dispatcher.py`. A scripted `post_completion` stub returns one segment per
`generate()` call, so no engine is involved.

| Test | Covers |
|---|---|
| `test_output_tokens_excludes_the_prompt` | the core defect |
| `test_output_tokens_is_empty_before_any_generation` | nothing generated yet, prompt present |
| `test_output_tokens_accumulates_across_generate_calls` | multi-segment concatenation |
| `test_zero_token_generation_leaves_output_tokens_unchanged` | empty segment / EOS |
| `test_output_tokens_survives_kv_modification` | the slicing trap in §3 |
| `test_output_accessors_agree_with_each_other` | the three accessors describe one span |
| `test_output_tokens_does_not_expose_mutable_internal_state` | returned list is a copy |
| `test_failed_generation_still_records_tokens_seen_before_the_error` | the `finally` guarantee |

Covers every case the issue called out (prompt-then-generate, consecutive
`generate()` calls, zero-token generation, generation after a KV modification)
plus the three invariants the fix introduces.

## 6. Verification

```console
$ pytest -q tests/sdk/test_request.py
8 passed

$ ruff check lmcache/sdk/request.py tests/sdk/test_request.py
All checks passed!

$ ruff format --check lmcache/sdk/request.py tests/sdk/test_request.py
2 files already formatted

$ mypy --config-file=pyproject.toml lmcache/sdk/request.py tests/sdk/test_request.py
Success: no issues found in 2 source files

$ codespell --toml pyproject.toml <changed files>
(clean)
```

**Regression tests confirmed meaningful** — reverting the property to
`return self.tokens` and re-running:

```
8 failed
```

All eight fail without the fix and pass with it.

### A real finding from the type check

`mypy` initially rejected the test stubs: they annotated the first parameter as
`Sequence[int]`, but the `PostCompletion` protocol declares `list[int]`, so the
stub did not satisfy the protocol. Fixed by matching the protocol signature
(parameter also renamed to `prompt_token_ids` to match). Worth noting because
`ruff` and `pytest` both passed cleanly while this was broken — the type check
was the only thing that caught it.

Carried as a **separate follow-up commit** rather than an amend + force-push,
since the branch was already published.

## 7. Compatibility

This changes a public property's return value. Anyone depending on the old
(docstring-contradicting) behaviour should read `.tokens` instead. Flagged
prominently in the PR description rather than buried, since silent behaviour
changes to public API are worse than noisy ones.

---

## 8. PR description

> **What this PR does / why we need it**:
>
> Fixes #4591.
>
> `LMCacheRequestStream.output_tokens` returned `self.tokens`, which is the live
> sequence backing the KV — the prompt plus everything generated. Its docstring
> promises "the concatenated generated tokens across all segments", and its two
> sibling accessors, `output_text` and `decoded_tokens`, both report generated
> content only. A caller pairing `output_tokens` with `output_text` therefore got
> two different spans, and `len(output_tokens) != decoded_tokens`.
>
> **Special notes for your reviewers**:
>
> The history deliberately is **not** derived by slicing `self.tokens` by prompt
> length. `update()` replaces that list wholesale when the KV is modified, so
> after a compaction the prompt prefix is no longer recoverable from it. Generated
> ids are instead accumulated in their own list alongside the existing
> `_text_parts`, extended in the same `finally` block, which keeps the history
> correct across KV edits and also preserves tokens already decoded when a
> generation fails partway.
>
> The property returns a copy so callers cannot mutate the recorded history.
>
> This is a behavioural change to a public property. Anyone currently relying on
> `output_tokens` returning the full sequence should read `.tokens` instead, which
> is unchanged.
>
> ```
> $ pytest -q tests/sdk/test_request.py
> 8 passed
> ```
>
> All 8 fail against `dev` and pass here, including the cases the issue called
> out: consecutive `generate()` calls, zero-token generation, and generation after
> a KV modification.
>
> **If applicable**:
>
> - [x] this PR contains user facing changes - docs added
> - [x] this PR contains unit tests
