"""Check stable Block and Chunk fingerprints exposed by parsers."""

from dataclasses import replace
import re

import pytest

from parsers import sha256_block, sha256_chunk
from schema import Block, Chunk, Kind


@pytest.mark.parametrize("record,digest", [
    (Block("doc", Kind.TABLE, headers=["Item"], rows=[["Café"]]), sha256_block),
    (Chunk("chunk", "doc", "Café"), sha256_chunk),
])
def test_digest_is_stable_and_tracks_changes(record, digest):
    first = replace(record, metadata={"page": 1, "nested": {"a": 2, "b": 3}})
    second = replace(record, metadata={"nested": {"b": 3, "a": 2}, "page": 1})
    original = digest(first)
    assert re.fullmatch(r"[0-9a-f]{64}", original)
    assert original == digest(second)
    assert original != digest(replace(first, text=first.text + " "))
    first.metadata["page"] = 2
    assert original != digest(first)


def test_table_row_order_matters():
    block = Block("doc", Kind.TABLE, headers=["Region"], rows=[["Europe"], ["Asia"]])
    assert sha256_block(block) != sha256_block(replace(block, rows=list(reversed(block.rows))))


@pytest.mark.parametrize("value,error", [(object(), TypeError), (float("nan"), ValueError)])
@pytest.mark.parametrize("record,digest", [
    (Block("doc", Kind.PARAGRAPH), sha256_block),
    (Chunk("chunk", "doc", "Text"), sha256_chunk),
])
def test_digest_rejects_unsupported_metadata(value, error, record, digest):
    with pytest.raises(error):
        digest(replace(record, metadata={"value": value}))
