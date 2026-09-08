# Vector DB operators and LanceDB

This package wraps **vector database backends** behind a small `VDB` interface (`adt_vdb.py`) and exposes two graph-style operators:

- **`IngestVdbOperator`** — writes embedded pipeline rows into a VDB (ingestion).
- **`RetrieveVdbOperator`** — runs similarity search given **precomputed query vectors** (retrieval).

The only built-in backend key today is **`lancedb`**, resolved by `get_vdb_op_cls()` in `factory.py` to the concrete **`LanceDB`** class in `lancedb.py`.

The root CLI is intentionally LanceDB-first: `retriever ingest ...` writes LanceDB tables, and `retriever query ...` queries LanceDB tables. Other VDB backends should plug in through the SDK/operator layer by implementing `VDB` and registering a backend key in `factory.py`; the root CLI does not expose a backend-agnostic VDB configuration surface.

---

## Collection capabilities

`VDB` defines required collection and document capabilities for the service API.
Backends implement the CRUD methods plus `write_collection()` and
`retrieve_collection()`; callers never pass logical collection identity through
legacy `run()` or `retrieval(**kwargs)`. Maintenance and health retain safe empty
defaults for backends without recoverable lifecycle work or additional health
details.

`CollectionWriteContext` carries immutable logical write identity. The service
and graph operators pass that context through unchanged; concrete backends own
physical names, schemas, native ranking fields, locks, and persistence. LanceDB
initializes its private collection catalog lazily, so ordinary fixed-table
construction and the existing CLI paths do not create collection metadata.

`VDB.stream_ingest(records)` is an optional, non-abstract capability for batch
ingest. It receives a lazy iterable of canonical NeMo Retriever Library record
dictionaries. The iterable does not contain Ray, pandas, Arrow, or
backend-specific objects. An implementation must consume it exactly once,
synchronously, and to exhaustion before returning; it must not retain the
iterable.
Existing `VDB` subclasses remain compatible because the inherited method raises
`UnsupportedVDBOperation`. `IngestVdbOperator` detects whether a backend
overrides the method before selecting the streaming path.

---

## `IngestVdbOperator` (ingestion)

### Role

`IngestVdbOperator` adapts **flat graph / DataFrame rows** (the shape produced after extract → embed in NeMo Retriever) into the canonical records expected by client VDBs. Legacy calls use **`VDB.run(records)`** once per batch; an explicit `CollectionWriteContext` dispatches to **`VDB.write_collection(records, context=...)`**.

Flow (see `operators/vdb.py` and `common/vdb/records.py`):

1. **`to_client_vdb_records(data)`** — converts rows to `list[list[dict]]` (one outer batch). Dense rows require an **embedding** plus either nonblank **text** or concrete image backing. Image-backed rows without text are stored as `type=image` and `text=""`; they are searchable through dense retrieval but add no FTS terms. The answer-oriented evidence formatter omits every hit without nonblank text and reports the omission in coverage. Sparse-only ingestion continues to require nonblank text.
2. Optional **sidecar metadata** — if `vdb_kwargs` contains `meta_dataframe` / `meta_source_field` / `meta_fields`, those keys are stripped for the concrete DB constructor and merged onto records via `sidecar_metadata.py`.
3. **Explicit dispatch** — calls `VDB.run(records)` for fixed-table ingestion or `VDB.write_collection(records, context=...)` for a scoped collection.

For streaming batch ingest, `IngestVdbOperator.stream_ingest(batches)` converts
each graph row lazily, applies the same sidecar metadata, and delegates one
canonical record iterable to `VDB.stream_ingest(records)`. Canonical conversion
stays in the operator, while the backend owns its write lifecycle.

### Ray batch pipelines (`RayDataExecutor`)

Graph ingestion with `run_mode=batch` uses **`RayDataExecutor`**
(`nemo_retriever/graph/executor.py`). `RayDataExecutor.ingest()` selects the
streaming path when the graph has one eligible `IngestVdbOperator` whose backend
overrides `VDB.stream_ingest()`.

On that path, the executor owns Ray batch iteration, prefetch, iterator cleanup,
retention of the historical pandas result, and downstream ordering. The
operator converts those batches into canonical records before the backend sees
them. The executor does not import LanceDB or pass Ray objects across the VDB
interface.

Backends that inherit the default `VDB.stream_ingest()` implementation retain
the historical global-batch path. `IngestVdbOperator.REQUIRES_GLOBAL_BATCH`
causes the complete dataset to be repartitioned to one block before
`VDB.run(records)` executes. `PutVdbOperator` explicitly opts out of streaming
so its update-only semantics remain on that path.

The public `RayDataExecutor.build_dataset()` method also retains its historical
behavior. It returns the full lazy graph, including the global VDB stage, and
does not perform streaming ingest. Call `RayDataExecutor.ingest()` to use the
streaming optimization and materialize the result.

In-process execution and service execution do not select this streaming path.
They continue through `IngestVdbOperator.process()` and its existing VDB
dispatch. Streaming selection is an automatic backend capability check, not a
generic ingest-time setting.

### Wiring ingestion today

- **Root CLI** (`retriever ingest ...`): writes to LanceDB through the shared graph ingest path.
- **Direct API**:

```python
from nemo_retriever.operators.vdb import IngestVdbOperator

op = IngestVdbOperator(
    vdb_op="lancedb",
    vdb_kwargs={
        "uri": "./kb",
        "table_name": "nemo-retriever",
        "vector_dim": 2048,
    },
)
op(pandas_dataframe_of_embedded_rows)  # or list of row dicts
```

The root CLI exposes the built-in LanceDB target directly:

```bash
retriever ingest /data/pdfs \
  --lancedb-uri ./kb \
  --table-name nemo-retriever
```

---

## LanceDB inside `IngestVdbOperator`

When `vdb_op="lancedb"` (or `vdb=LanceDB(...)` is passed explicitly), `_construct_vdb` instantiates **`LanceDB`** with the **clean** constructor kwargs (sidecar keys removed).

### LanceDB ingestion paths

`LanceDB.run` (in `lancedb.py`) remains the direct, in-process, and
legacy-fallback fixed-table ingestion path. It orchestrates:

1. **`create_index`** — connects with `lancedb.connect(self.uri)`, transforms ingestion batches into Arrow rows (`vector`, `text`, `metadata`, `source`), and **`db.create_table(...)`** with schema and `on_bad_vectors` policy.
2. **`write_to_index`** — builds the **vector index** (e.g. IVF/HNSW) and optionally an **FTS/BM25** index over the ingested `text` column when `hybrid=True`.

`LanceDB.stream_ingest` is the first-class bounded implementation of the
optional VDB capability. It receives canonical record dictionaries and owns
Arrow packing and schemas, the byte limit, one table mutation, durable operation
tags, retry and version validation, index coverage, and optional optimization.
Ray remains responsible only for producing and retaining ordered input batches.
Scheme-less local paths and authority-free `file:///` URIs also share a
crash-released table lock across backend instances and processes; other URIs
retain the legacy global-batch path.

Common constructor arguments include:

| Parameter        | Purpose |
|-----------------|--------|
| `uri`           | LanceDB database path/URI |
| `table_name`    | Table name (default `nemo-retriever`) |
| `overwrite`     | Table create mode vs append |
| `vector_dim`    | Expected embedding dimension (default 2048) |
| `index_type` / `metric` / `num_partitions` / `num_sub_vectors` | Vector index tuning |
| `hybrid`        | Also build the LanceDB FTS/BM25 index on ingested `text` |
| `on_bad_vectors`| `drop`, `fill`, `null`, or `error` |
| `stream_batch_bytes` | Maximum Arrow bytes per packed streaming batch (default 256 MiB) |
| `stream_optimize` | Run LanceDB optimization after a streaming write (default `False`) |
| `stream_operation_id` | Optional stable operation ID; generated and retained for failed same-instance retries when omitted; set explicitly for backend reconstruction or process restart |

---

## `RetrieveVdbOperator` (retrieval)

### Role

`RetrieveVdbOperator` wraps the same concrete **`VDB`** instance. Fixed-table calls use **`retrieval(vectors, **kwargs)`** and normalize legacy hit shapes; requests with both `scope` and `collection_name` use the explicit **`retrieve_collection(...)`** capability and validate/project its results into the canonical public hit contract. See `operators/vdb.py` and `common/vdb/records.py`.

Important: retrieval here expects **`vectors`** — a list of query embedding vectors — as the primary input. String queries are embedded elsewhere (e.g. in `Retriever`). Hybrid backends that need raw text receive aligned `query_texts` as execution-only call context.

Before embedding, `Retriever` asks the operator for `get_index_metadata("embedding_model_name")`. The base `VDB` implementation returns `None`; a backend can override the method to expose metadata from its selected table or index. LanceDB exposes both `embedding_model_name` and `retrieval_mode` through this lookup.

### LanceDB inside `RetrieveVdbOperator`

For `vdb_op="lancedb"`, **`LanceDB.retrieval`**:

- Opens the table with `lancedb.connect(table_path).open_table(table_name)`.
- For dense retrieval, each query vector uses **`table.search([vector], vector_column_name=..., **search_kwargs)`**, optional **`.where(where_clause)`** (Lance / DataFusion SQL; `metadata` / `source` are stored as JSON strings), then **`.limit(top_k).refine_factor(...).nprobes(...)`**.
- For hybrid retrieval, callers pass `hybrid=True` plus `query_texts` aligned with the vectors. LanceDB uses **`table.search(query_type="hybrid", vector_column_name=..., fts_columns="text").vector(vector).text(query_text)`** before applying the same `where`, limit, refine, probe, and select handling. Product query paths also pass the shared weighted-RRF policy (`candidate_depth=50`, `dense_weight=0.8`, `rrf_k=10`), then truncate the fused ranking to `top_k`. Direct low-level callers opt into that behavior explicitly with `hybrid_fusion=HybridFusionPolicy(...)`.

Notable kwargs: `top_k`, `refine_factor`, `n_probe` / `nprobes`, `where` or `_filter`, `table_path`, `table_name`, `search_kwargs`, `hybrid`, `query_texts`, and `hybrid_fusion`. `query_texts` is stripped from constructor kwargs and forwarded only for retrieval calls whose effective mode is hybrid.

Example of **direct** operator use (you supply vectors):

```python
from nemo_retriever.operators.vdb import RetrieveVdbOperator

op = RetrieveVdbOperator(
    vdb_op="lancedb",
    vdb_kwargs={"uri": "./kb", "table_name": "nemo-retriever"},
)
hits_per_query = op.process(
    [[0.1, 0.2, ...]],  # one query vector; dimension must match table
    top_k=5,
    where="metadata LIKE '%\"page_number\": 3%'",  # example; escape/quote for real SQL
)
```

---

## `Retriever` and `RetrieveVdbOperator`

The high-level **`Retriever`** class (`retriever.py`) uses **`RetrieveVdbOperator`** internally. Pass a flat LanceDB **`vdb_kwargs`** dict with `uri`, `table_name`, filters, etc., or the explicit nested shape `{"vdb_op": "lancedb", "vdb_kwargs": {...}}`.

For non-LanceDB backends, implement the `VDB` interface in a backend module, register the backend in `factory.py`, and construct `Retriever` through the SDK with `{"vdb_op": "<backend>", "vdb_kwargs": {...}}` or a concrete `{"vdb": backend_instance}`. The root `retriever query` CLI remains LanceDB-only.

It **lazy-builds** the operator:

```python
# Conceptually equivalent to:
RetrieveVdbOperator(vdb_op="lancedb", vdb_kwargs={**self.vdb_kwargs})
```

On **`query` / `queries`**, `Retriever`:

1. Embeds query text via the configured embedder (local HF or remote NIM).
2. Calls the retrieve operator’s **`process(vectors, ...)`** with merged **`vdb_kwargs`** (including per-call `where` / `_filter` for LanceDB).

Typical construction:

```python
from nemo_retriever.graph.retriever import Retriever

retriever = Retriever(
    vdb_kwargs={
        "uri": "./kb",
        "table_name": "nemo-retriever",
        "top_k": 10,
        "refine_factor": 50,
        "nprobes": 64,
    },
    embed_kwargs={
        "model_name": "nvidia/llama-nemotron-embed-1b-v2",
        "embed_model_name": "nvidia/llama-nemotron-embed-1b-v2",
    },
)
results = retriever.query("What is covered in section 2?")
```

Per-call Lance filters:

```python
retriever.query(
    "budget assumptions",
    vdb_kwargs={"where": "source LIKE '%annual_report%'", "top_k": 8},
)
```

---

## Metadata filtering

**Reference notebook:** [`examples/nemo_retriever_retriever_query_metadata_filter.ipynb`](../../../../examples/nemo_retriever_retriever_query_metadata_filter.ipynb) — runnable end-to-end demo using sidecar metadata and both filter modes below.

Two complementary mechanisms narrow `Retriever.query` results by metadata:

1. **Server-side (`where`)** — Pass a Lance / DataFusion SQL predicate in `vdb_kwargs` per call (or as a default on the `Retriever`). The predicate runs inside LanceDB on the table columns (`vector`, `text`, `metadata`, `source`) and is wired up in `LanceDB.retrieval` as a `.where(...)` clause on the vector search. **`_filter`** is accepted as an alias for `where`.
2. **Client-side** — Use **`filter_hits_by_content_metadata(hits, predicate)`** after retrieval to keep rows whose parsed `content_metadata` satisfies an arbitrary Python predicate. Useful for logic that doesn't fit SQL or for filters that depend on combined fields.

### How metadata is stored

During ingestion, each chunk's `content_metadata` is serialized as a **compact JSON string** (no spaces after `:` or `,`) in the `metadata` column of the LanceDB table. Sidecar columns supplied via `meta_dataframe` / `meta_source_field` / `meta_fields` are merged into that JSON object before upload — so sidecar keys live in the same JSON string, not in separate columns. This is why SQL filters on metadata use `LIKE` against a JSON substring rather than a real JSON operator.

### Writing `where` predicates

LanceDB evaluates `where` as DataFusion SQL. A few patterns:

```python
# Match a sidecar string field by exact value (compact JSON: "key":"value")
where = "metadata LIKE '%\"meta_a\":\"alpha\"%'"

# Match a numeric metadata field — numbers serialize without quotes
where = "metadata LIKE '%\"meta_b\":10%'"

# Combine predicates with AND / OR
where = "metadata LIKE '%\"meta_a\":\"bravo\"%' AND metadata LIKE '%\"meta_b\":10%'"

# Filter on the `source` column directly (separate from metadata JSON)
where = "source LIKE '%annual_report%'"
```

Escape single quotes in SQL strings by doubling them (`''`). Because matching is substring-based, include the JSON key (`"meta_a":` rather than just `alpha`) to avoid matching unrelated values.

### Server-side vs client-side

Use **`where`** when the predicate fits SQL and you want LanceDB to prune candidates before vector ranking — it also avoids the wasted work of materializing hits you'd discard. Use **`filter_hits_by_content_metadata`** when the predicate is easier to express in Python (e.g. combined numeric ranges, membership in a Python set, or fields that need parsing). They compose well — run a wide `top_k` with a `where` to prune broadly, then post-filter client-side for finer logic:

```python
from nemo_retriever.common.vdb.sidecar_metadata import filter_hits_by_content_metadata

hits = retriever.query(
    "budget assumptions",
    top_k=16,
    vdb_kwargs={"where": "metadata LIKE '%\"meta_a\":\"bravo\"%'"},
)
hits = filter_hits_by_content_metadata(
    hits, lambda m: m.get("meta_b", 0) >= 10
)
```

### Inspecting hit metadata

Each hit's `metadata` field is a JSON string. Use **`parse_hit_content_metadata(hit)`** to get a `dict` you can read directly (this is what `filter_hits_by_content_metadata` uses internally). Both helpers are exported from `nemo_retriever.common.vdb`.

### Hybrid retrieval

Hybrid search (`hybrid=True`) is implemented for LanceDB's precomputed-vector retrieval path. It requires `query_texts` aligned one-to-one with the query vectors so the backend can combine the dense vector query with full-text search. Filters above apply to both dense and hybrid search.

---

## End-to-end mental model

```mermaid
flowchart LR
  subgraph ingest
    G[Graph rows / DataFrame]
    IVO[IngestVdbOperator]
    R1[to_client_vdb_records]
    L1[LanceDB.run / stream_ingest]
    G --> IVO --> R1 --> L1
  end

  subgraph retrieve
    Q[Query strings]
    E[Embed queries]
    RVO[RetrieveVdbOperator]
    L2[LanceDB.retrieval]
    Q --> E --> RVO --> L2
  end

  L1 -->[(LanceDB table on disk)]
  L2 -->[(same table)]
```

- **Ingest**: flat rows → canonical records → **`LanceDB.run`** or **`LanceDB.stream_ingest`** → table + indexes.
- **Retrieve**: strings → vectors → **`RetrieveVdbOperator`** → **`LanceDB.retrieval`** → hit lists.

For implementation details, see `operators/vdb.py`, `adt_vdb.py`, `lancedb.py`, `sink.py`, `records.py`, `factory.py`, and `retriever.py`.
