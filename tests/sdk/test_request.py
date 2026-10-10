# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LMCacheRequestStream's generation-history accessors.

Covers the documented contract that output_tokens / output_text /
decoded_tokens report only generated content, across multiple generate()
calls and across KV modifications that replace the live sequence.
"""

# Standard
from collections.abc import Iterable, Sequence
from typing import Any
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.sdk.cache_kind import LMCacheSDKCacheKind
from lmcache.sdk.request import LMCacheRequestStream, TokenEvent

PROMPT = [101, 102, 103]


def _post_completion(*segments: Sequence[tuple[int, str]]):
    """Build a post_completion yielding one scripted segment per call.

    Args:
        segments: One (token_id, text) sequence per expected generate() call.

    Returns:
        A PostCompletion-compatible callable.
    """
    remaining = list(segments)

    def post_completion(
        prompt_token_ids: list[int], sampling_params: dict[str, Any], cache_salt: str
    ) -> Iterable[TokenEvent]:
        return [TokenEvent(token_id=tid, text=text) for tid, text in remaining.pop(0)]

    return post_completion


def _kv_context() -> MagicMock:
    ctx = MagicMock(name="kv_context")
    ctx.kind = LMCacheSDKCacheKind.KV
    return ctx


def _stream(*segments: Sequence[tuple[int, str]]) -> LMCacheRequestStream:
    return LMCacheRequestStream(
        contexts=[_kv_context()],
        post_completion=_post_completion(*segments),
        prompt_token_ids=PROMPT,
    )


def test_output_tokens_excludes_the_prompt() -> None:
    """output_tokens reports generated ids only, while tokens keeps the prompt."""
    stream = _stream([(201, "a"), (202, "b")])

    stream.generate({"max_tokens": 2})

    assert stream.output_tokens == [201, 202]
    assert stream.tokens == PROMPT + [201, 202]


def test_output_tokens_is_empty_before_any_generation() -> None:
    """A fresh stream has generated nothing, even though tokens holds the prompt."""
    stream = _stream()

    assert stream.output_tokens == []
    assert stream.tokens == PROMPT


def test_output_tokens_accumulates_across_generate_calls() -> None:
    """Consecutive segments concatenate in generation order."""
    stream = _stream([(201, "a"), (202, "b")], [(203, "c")])

    stream.generate({"max_tokens": 2})
    stream.generate({"max_tokens": 2})

    assert stream.output_tokens == [201, 202, 203]


def test_zero_token_generation_leaves_output_tokens_unchanged() -> None:
    """An empty segment contributes nothing and marks the stream done."""
    stream = _stream([(201, "a")], [])

    stream.generate({"max_tokens": 1})
    stream.generate({"max_tokens": 1})

    assert stream.output_tokens == [201]
    assert stream.is_done


def test_output_tokens_survives_kv_modification() -> None:
    """update() replaces the live sequence but must not drop generation history.

    This is why the history cannot be derived by slicing tokens: after a
    compaction the prompt prefix is no longer recoverable from tokens.
    """
    stream = _stream([(201, "a"), (202, "b")], [(203, "c")])
    stream.generate({"max_tokens": 2})

    stream.update(LMCacheSDKCacheKind.KV, torch.zeros(2, 1, 2, 4), [999, 998])
    assert stream.tokens == [999, 998]

    stream.generate({"max_tokens": 2})

    assert stream.output_tokens == [201, 202, 203]
    assert stream.tokens == [999, 998, 203]


def test_output_accessors_agree_with_each_other() -> None:
    """decoded_tokens, output_text and output_tokens describe the same segments."""
    stream = _stream([(201, "he"), (202, "llo")], [(203, "!")])

    stream.generate({"max_tokens": 2})
    stream.generate({"max_tokens": 2})

    assert stream.output_text == "hello!"
    assert stream.decoded_tokens == 3
    assert len(stream.output_tokens) == stream.decoded_tokens


def test_output_tokens_does_not_expose_mutable_internal_state() -> None:
    """Mutating the returned list must not corrupt the recorded history."""
    stream = _stream([(201, "a")])
    stream.generate({"max_tokens": 1})

    stream.output_tokens.append(999)

    assert stream.output_tokens == [201]


def test_failed_generation_still_records_tokens_seen_before_the_error() -> None:
    """Partial segments are retained so the history matches what was decoded."""

    def post_completion(
        prompt_token_ids: list[int], sampling_params: dict[str, Any], cache_salt: str
    ) -> Iterable[TokenEvent]:
        yield TokenEvent(token_id=201, text="a")
        raise RuntimeError("engine died")

    stream = LMCacheRequestStream(
        contexts=[_kv_context()],
        post_completion=post_completion,
        prompt_token_ids=PROMPT,
    )

    with pytest.raises(Exception, match="failed during generation"):
        stream.generate({"max_tokens": 2})

    assert stream.output_tokens == [201]
    assert stream.decoded_tokens == 1
