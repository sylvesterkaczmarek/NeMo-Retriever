# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

lancedb = pytest.importorskip("lancedb", minversion="0.34.0")

from nemo_retriever.common.vdb.lancedb import LanceDB
from nemo_retriever.common.vdb.sink import OversizedVdbRowError, VdbWriteNotFinalized
from nemo_retriever.common.vdb.sink_operation import CommitOutcomeUnknown, VdbOperationConflict


def _record(
    row_id: int,
    *,
    vector_dim: int = 2,
    embedding: list[float] | None = None,
    text_prefix: str = "chunk",
    padding: int = 0,
) -> dict[str, Any]:
    vector = embedding if embedding is not None else [float(row_id), *[1.0] * (vector_dim - 1)]
    return {
        "document_type": "text",
        "metadata": {
            "embedding": vector,
            "content": f"{text_prefix}-{row_id} " + ("x" * padding),
            "content_metadata": {
                "type": "text",
                "id": f"row-{row_id}",
                "page_number": row_id,
            },
            "source_metadata": {
                "source_id": f"/tmp/doc-{row_id}.pdf",
                "source_name": f"doc-{row_id}.pdf",
            },
        },
    }


def _records(
    start: int,
    stop: int,
    *,
    vector_dim: int = 2,
    text_prefix: str = "chunk",
    padding: int = 0,
) -> list[dict[str, Any]]:
    return [
        _record(
            row_id,
            vector_dim=vector_dim,
            text_prefix=text_prefix,
            padding=padding,
        )
        for row_id in range(start, stop)
    ]


def _backend(uri: Path, **overrides: Any) -> LanceDB:
    kwargs: dict[str, Any] = {
        "uri": str(uri),
        "table_name": "chunks",
        "vector_dim": 2,
        "overwrite": True,
        "build_index": False,
        "stream_batch_bytes": 256 << 20,
    }
    kwargs.update(overrides)
    return LanceDB(**kwargs)


def _table(uri: Path):
    table = lancedb.connect(str(uri)).open_table("chunks")
    table.checkout_latest()
    return table


def _state(uri: Path) -> tuple[list[str], list[int]]:
    table = _table(uri)
    ids = sorted(table.to_arrow().column("id").to_pylist())
    versions = [int(version["version"]) for version in table.list_versions()]
    return ids, versions


def _product_metadata(schema: pa.Schema) -> dict[bytes, bytes]:
    return {key: value for key, value in (schema.metadata or {}).items() if not key.startswith(b"nemo_retriever.sink_")}


def test_stream_ingest_is_lazy_and_byte_bounded_with_legacy_query_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical records become bounded Arrow batches in one indexed lifecycle."""

    common = {
        "build_index": True,
        "hybrid": True,
        "index_type": "IVF_FLAT",
        "num_partitions": 2,
    }
    legacy = _backend(tmp_path / "legacy", **common)
    streaming = _backend(
        tmp_path / "streaming",
        stream_batch_bytes=1024,
        stream_operation_id="indexed-overwrite",
        **common,
    )
    records = _records(0, 16, padding=200)
    legacy.run([records])

    pulls: list[int] = []
    observed_batch_bytes: list[int] = []
    finalized_row_counts: list[int] = []

    connection_type = type(lancedb.connect(str(tmp_path / "streaming")))
    original_create_table = connection_type.create_table

    def observed_create_table(self, name, data=None, *args, **kwargs):
        if name == "chunks":
            assert isinstance(data, pa.RecordBatchReader)
            assert 0 < len(pulls) < len(records)
            original_reader = data

            def observed_batches() -> Iterator[pa.RecordBatch]:
                for batch in original_reader:
                    assert isinstance(batch, pa.RecordBatch)
                    observed_batch_bytes.append(int(batch.get_total_buffer_size()))
                    yield batch

            data = pa.RecordBatchReader.from_batches(original_reader.schema, observed_batches())
        return original_create_table(self, name, data=data, *args, **kwargs)

    monkeypatch.setattr(connection_type, "create_table", observed_create_table)
    original_maintain_indexes = streaming._maintain_indexes

    def observe_finalization(records, table):
        finalized_row_counts.append(int(table.count_rows()))
        return original_maintain_indexes(records, table)

    monkeypatch.setattr(streaming, "_maintain_indexes", observe_finalization)

    def record_stream() -> Iterator[dict[str, Any]]:
        for row_index, record in enumerate(records):
            pulls.append(row_index)
            yield record

    streaming.stream_ingest(record_stream())

    assert pulls == list(range(len(records)))
    assert len(observed_batch_bytes) > 1
    assert max(observed_batch_bytes) <= 1024
    assert finalized_row_counts == [len(records)]

    legacy_table = _table(tmp_path / "legacy")
    streaming_table = _table(tmp_path / "streaming")
    assert legacy_table.schema.remove_metadata() == streaming_table.schema.remove_metadata()
    assert _product_metadata(legacy_table.schema) == _product_metadata(streaming_table.schema)
    assert legacy_table.to_arrow().sort_by("id").to_pylist() == streaming_table.to_arrow().sort_by("id").to_pylist()
    assert {tuple(index.columns) for index in streaming_table.list_indices()} == {
        ("vector",),
        ("text",),
    }

    vectors = [[1.0, 1.0], [13.0, 1.0]]
    query_texts = ["chunk-1", "chunk-13"]
    legacy_hits = legacy.retrieval(vectors, query_texts=query_texts, hybrid=True, top_k=3)
    streaming_hits = streaming.retrieval(vectors, query_texts=query_texts, hybrid=True, top_k=3)
    assert [[hit["id"] for hit in hits] for hits in streaming_hits] == [
        [hit["id"] for hit in hits] for hits in legacy_hits
    ]


def test_oversized_row_fails_without_table_mutation(tmp_path: Path) -> None:
    backend = _backend(
        tmp_path,
        stream_batch_bytes=256,
        stream_operation_id="oversized",
    )

    with pytest.raises(OversizedVdbRowError, match="max_batch_bytes=256"):
        backend.stream_ingest([_record(0, padding=4096)])

    assert "chunks" not in lancedb.connect(str(tmp_path)).list_tables().tables


@pytest.mark.parametrize(
    "control",
    [
        {"stream_batch_bytes": 1024},
        {"stream_optimize": True},
        {"stream_operation_id": "fixed-operation"},
    ],
)
def test_stream_controls_are_rejected_by_legacy_mutations(tmp_path: Path, control: dict[str, Any]) -> None:
    backend = _backend(tmp_path, **control)
    message = r"require LanceDB\.stream_ingest\(\)"

    with pytest.raises(ValueError, match=message):
        backend.run([])
    with pytest.raises(ValueError, match=message):
        backend.put([])


def test_stream_refreshes_cached_table_for_legacy_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend(tmp_path, overwrite=False)
    backend.run([_records(0, 1)])
    assert backend._open_table("chunks").count_rows() == 1

    backend.stream_ingest(_records(1, 2))
    assert backend._open_table("chunks").count_rows() == 2

    class ArrowTableView:
        def __init__(self, table) -> None:
            self.table = table

        def to_table(self, *, columns, filter):
            return self.table.to_arrow().select(columns)

    table_type = type(backend._open_table("chunks"))
    monkeypatch.setattr(table_type, "to_lance", lambda table, **kwargs: ArrowTableView(table))
    counts = backend.put([[_record(1, text_prefix="updated")]])
    backend.run([_records(2, 3)])

    rows = {row["id"]: row for row in backend._open_table("chunks").to_arrow().to_pylist()}
    assert counts["put"] == 1
    assert set(rows) == {"row-0", "row-1", "row-2"}
    assert rows["row-1"]["text"].startswith("updated-1")


@pytest.mark.parametrize(
    ("policy", "embedding", "expected_vectors"),
    [
        pytest.param("drop", [1.0], [], id="drop-wrong-width"),
        pytest.param("fill", [1.0], [[-3.5, -3.5]], id="fill-wrong-width"),
        pytest.param("fill", [1.0, float("nan")], [[-3.5, -3.5]], id="fill-nan"),
        pytest.param("null", [1.0], [None], id="null-wrong-width"),
    ],
)
def test_bad_vector_policies_match_legacy(
    tmp_path: Path,
    policy: str,
    embedding: list[float],
    expected_vectors: list[list[float] | None],
) -> None:
    common = {
        "validate_vector_length": False,
        "on_bad_vectors": policy,
        "fill_value": -3.5,
    }
    legacy = _backend(tmp_path / "legacy", **common)
    streaming = _backend(
        tmp_path / "streaming",
        stream_operation_id=f"bad-vector-{policy}",
        **common,
    )

    legacy.run([[_record(0, embedding=embedding)]])
    streaming.stream_ingest([_record(0, embedding=embedding)])

    legacy_vectors = _table(tmp_path / "legacy").to_arrow()["vector"].to_pylist()
    streaming_vectors = _table(tmp_path / "streaming").to_arrow()["vector"].to_pylist()
    assert streaming_vectors == legacy_vectors == expected_vectors


def test_deferred_dimension_inference_after_invalid_prefix_matches_legacy(tmp_path: Path) -> None:
    records = [
        _record(0, embedding=[]),
        _record(1, embedding=[float("nan"), 1.0, 2.0]),
        _record(2, embedding=[2.0, 1.0, 2.0]),
    ]
    common = {"vector_dim": None, "on_bad_vectors": "drop"}
    legacy = _backend(tmp_path / "legacy", **common)
    streaming = _backend(
        tmp_path / "streaming",
        stream_operation_id="infer-after-invalid-prefix",
        **common,
    )

    legacy.run([records])
    streaming.stream_ingest(iter(records))

    legacy_table = _table(tmp_path / "legacy")
    streaming_table = _table(tmp_path / "streaming")
    assert streaming_table.schema.field("vector").type.list_size == 3
    assert streaming_table.to_arrow().to_pylist() == legacy_table.to_arrow().to_pylist()
    assert streaming_table.to_arrow().column("id").to_pylist() == ["row-2"]


def test_append_preserves_compatible_existing_schema(tmp_path: Path) -> None:
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), 2)),
            pa.field("text", pa.string()),
            pa.field("metadata", pa.string()),
            pa.field("source", pa.string()),
            pa.field("id", pa.string()),
            pa.field("legacy_extra", pa.string()),
        ],
        metadata={
            b"retrieval_mode": b"dense",
            b"nemo_retriever.retrieval_mode": b"dense",
        },
    )
    db = lancedb.connect(str(tmp_path))
    db.create_table(
        "chunks",
        data=[
            {
                "vector": [1.0, 0.0],
                "text": "seed",
                "metadata": '{"type":"text","id":"seed"}',
                "source": '{"source_id":"/tmp/seed.pdf"}',
                "id": "seed",
                "legacy_extra": "preserved",
            }
        ],
        schema=schema,
    )

    _backend(
        tmp_path,
        overwrite=False,
        stream_operation_id="append-schema-superset",
    ).stream_ingest([_record(1)])

    table = _table(tmp_path)
    rows = sorted(table.to_arrow().to_pylist(), key=lambda row: row["id"])
    assert table.schema == schema
    assert [(row["id"], row["legacy_extra"]) for row in rows] == [
        ("row-1", None),
        ("seed", "preserved"),
    ]


def _fail_after_first_arrow_batch() -> Iterator[dict[str, Any]]:
    yield _record(10, padding=64)
    yield _record(11, padding=64)
    raise RuntimeError("injected source failure")


@pytest.mark.parametrize("overwrite", [True, False], ids=["overwrite", "append"])
def test_midstream_failure_preserves_target_and_retry_is_exactly_once(
    tmp_path: Path,
    overwrite: bool,
) -> None:
    _backend(
        tmp_path,
        stream_operation_id="seed",
    ).stream_ingest(_records(0, 2))
    before = _state(tmp_path)
    backend = _backend(
        tmp_path,
        overwrite=overwrite,
        stream_batch_bytes=512,
        stream_operation_id=f"write-{overwrite}",
    )

    with pytest.raises(RuntimeError, match="injected source failure"):
        backend.stream_ingest(_fail_after_first_arrow_batch())

    assert _state(tmp_path) == before
    backend.stream_ingest(_records(10, 14, padding=64))

    ids, versions = _state(tmp_path)
    expected = [f"row-{row_id}" for row_id in range(10, 14)]
    if not overwrite:
        expected.extend(["row-0", "row-1"])
    assert ids == sorted(expected)
    assert len(ids) == len(set(ids))
    assert len(versions) == len(before[1]) + 1


def test_acknowledged_append_retry_is_a_noop_and_changed_content_conflicts(tmp_path: Path) -> None:
    _backend(
        tmp_path,
        stream_operation_id="seed",
    ).stream_ingest(_records(0, 2))
    backend = _backend(
        tmp_path,
        overwrite=False,
        stream_operation_id="append-10-14",
    )
    backend.stream_ingest(_records(10, 14))
    after_first = _state(tmp_path)

    backend.stream_ingest(iter(_records(10, 14)))
    assert _state(tmp_path) == after_first

    with pytest.raises(VdbOperationConflict, match="different canonical content"):
        backend.stream_ingest(_records(10, 14, text_prefix="changed"))
    assert _state(tmp_path) == after_first


def test_default_operation_id_recovers_finalization_and_clears_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _backend(
        tmp_path,
        stream_operation_id="seed",
    ).stream_ingest(_records(0, 2))
    backend = _backend(
        tmp_path,
        overwrite=False,
        build_index=True,
        index_type="IVF_FLAT",
        num_partitions=2,
    )
    original_write_to_index = backend.write_to_index

    def fail_index(*args, **kwargs):
        raise RuntimeError("injected index failure")

    monkeypatch.setattr(backend, "write_to_index", fail_index)
    with pytest.raises(RuntimeError, match="injected index failure") as failure:
        backend.stream_ingest(_records(10, 14))
    assert any("LanceDB stream operation_id:" in note for note in failure.value.__notes__)
    with pytest.raises(ValueError, match="pending stream recovery"):
        backend.run([])
    with pytest.raises(ValueError, match="pending stream recovery"):
        backend.put([])

    reconstructed = _backend(tmp_path, overwrite=False)
    with pytest.raises(VdbWriteNotFinalized, match="original stream_operation_id"):
        reconstructed.run([])
    with pytest.raises(VdbWriteNotFinalized, match="original stream_operation_id"):
        reconstructed.put([])

    assert _state(tmp_path)[0] == [
        "row-0",
        "row-1",
        "row-10",
        "row-11",
        "row-12",
        "row-13",
    ]
    with pytest.raises(VdbWriteNotFinalized, match="not finalized"):
        backend.retrieval([[0.0, 1.0]], top_k=10)
    with pytest.raises(VdbOperationConflict, match="unfinished bounded-sink operation"):
        _backend(
            tmp_path,
            overwrite=False,
            stream_operation_id="different-append",
        ).stream_ingest(_records(20, 22))

    monkeypatch.setattr(backend, "write_to_index", original_write_to_index)
    table_type = type(_table(tmp_path))
    original_add = table_type.add

    def unexpected_add(*args, **kwargs):
        pytest.fail("retry re-added rows after the data mutation was acknowledged")

    monkeypatch.setattr(table_type, "add", unexpected_add)
    backend.stream_ingest(iter(_records(10, 14)))

    assert _state(tmp_path)[0] == [
        "row-0",
        "row-1",
        "row-10",
        "row-11",
        "row-12",
        "row-13",
    ]
    assert backend.retrieval([[0.0, 1.0]], top_k=10)

    monkeypatch.setattr(table_type, "add", original_add)
    backend.stream_ingest(_records(20, 22))
    assert _state(tmp_path)[0][-2:] == ["row-20", "row-21"]


def test_lost_append_commit_acknowledgement_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _backend(
        tmp_path,
        stream_operation_id="seed",
    ).stream_ingest(_records(0, 2))
    backend = _backend(
        tmp_path,
        overwrite=False,
        stream_operation_id="append-unknown",
    )
    tags_type = type(_table(tmp_path).tags)
    original_create = tags_type.create

    def fail_data_marker(self, tag: str, version: int) -> None:
        if tag.startswith("nemo_sink_data_"):
            raise RuntimeError("injected lost acknowledgement")
        original_create(self, tag, version)

    monkeypatch.setattr(tags_type, "create", fail_data_marker)
    with pytest.raises(RuntimeError, match="injected lost acknowledgement"):
        backend.stream_ingest(_records(10, 14))
    after_commit = _state(tmp_path)
    monkeypatch.setattr(tags_type, "create", original_create)

    with pytest.raises(CommitOutcomeUnknown, match="refusing to replay append"):
        backend.stream_ingest(_records(10, 14))
    assert _state(tmp_path) == after_commit
