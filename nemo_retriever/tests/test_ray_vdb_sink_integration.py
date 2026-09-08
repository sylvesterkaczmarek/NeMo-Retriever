# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Ray and LanceDB coverage for streaming VDB ingestion."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

ray = pytest.importorskip("ray", minversion="2.56.1")
lancedb = pytest.importorskip("lancedb", minversion="0.34.0")

from nemo_retriever.graph.executor import RayDataExecutor
from nemo_retriever.graph.pipeline_graph import Graph
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.vdb import IngestVdbOperator

_GRAPH_SCHEMA = pa.schema(
    [
        pa.field("block_id", pa.int64()),
        pa.field("text", pa.string()),
        pa.field(
            "text_embeddings_1b_v2",
            pa.struct([pa.field("embedding", pa.list_(pa.float32()))]),
        ),
        pa.field("source_id", pa.string()),
        pa.field("page_number", pa.int64()),
        pa.field(
            "metadata",
            pa.struct(
                [
                    pa.field(
                        "content_metadata",
                        pa.struct(
                            [
                                pa.field("type", pa.string()),
                                pa.field("id", pa.string()),
                            ]
                        ),
                    )
                ]
            ),
        ),
        pa.field("result_only", pa.string()),
    ]
)


def _source_table(block_id: int) -> pa.Table:
    row_ids = range(block_id * 2, block_id * 2 + 2)
    return pa.Table.from_pylist(
        [
            {
                "block_id": block_id,
                "text": f"chunk-{row_id}",
                "text_embeddings_1b_v2": {"embedding": [float(row_id), 1.0]},
                "source_id": f"/tmp/doc-{row_id}.pdf",
                "page_number": row_id,
                "metadata": {
                    "content_metadata": {
                        "type": "text",
                        "id": f"row-{row_id}",
                    }
                },
                "result_only": f"not-stored-{row_id}",
            }
            for row_id in row_ids
        ],
        schema=_GRAPH_SCHEMA,
    )


def _observe_lance_batches(data: Any, *, pulled_rows: list[int]) -> Iterable[pa.RecordBatch]:
    """Observe the real RecordBatch stream without replacing LanceDB mutation."""

    assert not isinstance(data, (list, tuple, pd.DataFrame, pa.Table))
    for batch in data:
        assert isinstance(batch, pa.RecordBatch)
        pulled_rows.append(batch.num_rows)
        yield batch


@pytest.mark.integration
def test_ray_streams_three_blocks_into_real_lancedb_and_preserves_contract(tmp_path, monkeypatch) -> None:
    """Streaming ingest avoids the global sink barrier and executes in order once."""

    assert ray.__version__ == "2.56.1"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=4, num_gpus=0, include_dashboard=False, log_to_driver=False)

    try:

        class _CompletedBlocks:
            def __init__(self) -> None:
                self._completed_blocks: list[int] = []

            def record_completed_block(self, block_id: int) -> None:
                self._completed_blocks.append(block_id)

            def completed_blocks(self) -> list[int]:
                return list(self._completed_blocks)

        def record_completed_block(batch: pa.Table, completed: Any) -> pa.Table:
            block_id = int(batch.column("block_id")[0].as_py())
            ray.get(completed.record_completed_block.remote(block_id))
            return batch

        completed_type = ray.remote(_CompletedBlocks)
        completed = completed_type.options(num_cpus=0).remote()
        source = ray.data.from_arrow([_source_table(i) for i in range(3)], override_num_blocks=3)
        assert source.num_blocks() == 3
        dataset = source.map_batches(
            record_completed_block,
            batch_format="pyarrow",
            batch_size=None,
            concurrency=3,
            fn_kwargs={"completed": completed},
        )

        class RequireFinalizedWrite(AbstractOperator):
            def preprocess(self, data, **kwargs):
                return data

            def process(self, data, **kwargs):
                table = lancedb.connect(str(tmp_path)).open_table("chunks")
                assert table.count_rows() == 6
                result = data.copy()
                result["after_sink"] = True
                return result

            def postprocess(self, data, **kwargs):
                return data

        graph = (
            Graph()
            >> IngestVdbOperator(
                vdb_op="lancedb",
                vdb_kwargs={
                    "uri": str(tmp_path),
                    "table_name": "chunks",
                    "vector_dim": 2,
                    "overwrite": True,
                    "build_index": False,
                    "stream_batch_bytes": 512,
                    "stream_operation_id": "ray-three-blocks",
                },
            )
            >> RequireFinalizedWrite()
        )
        executor = RayDataExecutor(graph)

        lazy = executor.build_dataset(dataset)
        assert isinstance(lazy, ray.data.Dataset)
        assert lancedb.connect(str(tmp_path)).list_tables().tables == []

        original_repartition = ray.data.Dataset.repartition

        def reject_global_repartition(self, *args, **kwargs):
            requested_blocks = kwargs.get("num_blocks", args[0] if args else None)
            if requested_blocks == 1:
                raise AssertionError("streaming ingest must not call repartition(num_blocks=1)")
            return original_repartition(self, *args, **kwargs)

        monkeypatch.setattr(ray.data.Dataset, "repartition", reject_global_repartition)

        original_iter_batches = ray.data.Dataset.iter_batches
        prefetch_calls: list[int | None] = []
        iterator_closed = False

        class ClosingIterator:
            def __init__(self, inner) -> None:
                self._inner = iter(inner)

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._inner)

            def close(self) -> None:
                nonlocal iterator_closed
                iterator_closed = True
                close = getattr(self._inner, "close", None)
                if callable(close):
                    close()

        def observe_first_iter_batches(self, *args, **kwargs):
            iterator = original_iter_batches(self, *args, **kwargs)
            if not prefetch_calls:
                prefetch_calls.append(kwargs.get("prefetch_batches"))
                return ClosingIterator(iterator)
            return iterator

        monkeypatch.setattr(ray.data.Dataset, "iter_batches", observe_first_iter_batches)

        connection_type = type(lancedb.connect(str(tmp_path)))
        original_create_table = connection_type.create_table
        pulled_rows: list[int] = []

        def observed_create_table(self, name, data=None, *args, **kwargs):
            if name == "chunks" and data is not None:
                data = _observe_lance_batches(data, pulled_rows=pulled_rows)
            return original_create_table(self, name, data, *args, **kwargs)

        monkeypatch.setattr(connection_type, "create_table", observed_create_table)

        result = executor.ingest(dataset)

        assert Counter(ray.get(completed.completed_blocks.remote())) == Counter({0: 1, 1: 1, 2: 1})
        assert prefetch_calls == [1]
        assert iterator_closed is True
        assert sum(pulled_rows) == 6

        result = result.sort_values("page_number", ignore_index=True)
        assert result["page_number"].tolist() == list(range(6))
        assert result["text"].tolist() == [f"chunk-{row_id}" for row_id in range(6)]
        assert result["result_only"].tolist() == [f"not-stored-{row_id}" for row_id in range(6)]
        assert result["after_sink"].tolist() == [True] * 6

        stored = lancedb.connect(str(tmp_path)).open_table("chunks").to_arrow().sort_by("id")
        assert stored.column_names == ["vector", "text", "metadata", "source", "id"]
        assert stored.column("id").to_pylist() == [f"row-{row_id}" for row_id in range(6)]
    finally:
        ray.shutdown()
