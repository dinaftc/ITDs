-- Relational schema of an interaction trace database.
-- {dim} is replaced by the embedding dimension; all objects live in the
-- schema selected by the connection's search_path (one schema per ITD).

CREATE TABLE IF NOT EXISTS task (
    task_id        TEXT PRIMARY KEY,
    task_family    TEXT NOT NULL,
    template_desc  TEXT NOT NULL,
    task_desc      TEXT NOT NULL
);

-- One row per task template; similarity retrieval ranks templates, not traces.
CREATE TABLE IF NOT EXISTS template (
    task_family    TEXT NOT NULL,
    template_desc  TEXT NOT NULL,
    embedding      vector({dim}) NOT NULL,
    PRIMARY KEY (task_family, template_desc)
);
CREATE INDEX IF NOT EXISTS template_hnsw_idx
    ON template USING hnsw (embedding vector_cosine_ops);

-- task_family and template_desc are materialized in trace because Maintain
-- and Retrieve filter on them.
CREATE TABLE IF NOT EXISTS trace (
    trace_id          UUID PRIMARY KEY,
    task_id           TEXT NOT NULL REFERENCES task,
    task_family       TEXT NOT NULL,
    template_desc     TEXT NOT NULL,
    actor             TEXT NOT NULL,
    label             TEXT NOT NULL CHECK (label IN ('success', 'failure')),
    reward            DOUBLE PRECISION NOT NULL,
    horizon           INTEGER NOT NULL,
    quality_category  TEXT NOT NULL CHECK (quality_category IN ('success', 'near-success', 'failure')),
    quality_score     DOUBLE PRECISION NOT NULL,
    seq               BIGSERIAL            -- arrival order; last tie-breaker
);
-- Partition lookup and in-partition ranking.
CREATE INDEX IF NOT EXISTS trace_partition_idx
    ON trace (task_family, template_desc, quality_category, quality_score DESC);
CREATE INDEX IF NOT EXISTS trace_task_idx ON trace (task_id);

CREATE TABLE IF NOT EXISTS step (
    trace_id     UUID NOT NULL REFERENCES trace ON DELETE CASCADE,
    step_index   INTEGER NOT NULL,
    observation  TEXT NOT NULL,
    action       TEXT NOT NULL,
    PRIMARY KEY (trace_id, step_index)
);

-- Materialized experiences, cached by task template, reuse strategy and synthesizer.
CREATE TABLE IF NOT EXISTS experience (
    task_family    TEXT NOT NULL,
    template_desc  TEXT NOT NULL,
    strategy       TEXT NOT NULL,
    synthesizer    TEXT NOT NULL DEFAULT '',
    content        TEXT NOT NULL,
    trace_ids      UUID[] NOT NULL,
    "timestamp"    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (task_family, template_desc, strategy, synthesizer)
);
