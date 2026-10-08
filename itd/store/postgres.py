"""PostgreSQL + pgvector instantiation (Sections 3.1 and 4.8).

B-tree index on (task_family, template_desc, quality_category, quality_score)
for partition filtering and trace ranking; HNSW index on template embeddings
for similarity fallback. One PostgreSQL schema per ITD.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row

from ..model import Experience, Step, Task, Trace
from .base import EXACT_FALLBACK, ITDStore

_TRACE_COLS = ("trace_id, task_id, task_family, template_desc, actor, label, reward, horizon, "
               "quality_category, quality_score")
_ORDER = "quality_score DESC, horizon ASC, seq ASC"


class PostgresStore(ITDStore):
    def __init__(self, embedder, dsn: str, namespace: str = "itd", *, reset: bool = False,
                 fallback_templates: int = 20):
        super().__init__(embedder)
        self.namespace = namespace
        self.fallback_templates = fallback_templates
        self.conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
        schema = sql.Identifier(namespace)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        if reset:
            self.conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(schema))
        self.conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema))
        self.conn.execute(sql.SQL("SET search_path TO {}, public").format(schema))
        # Keep scanning the HNSW graph until the task_family filter is satisfied (pgvector >= 0.8).
        self.conn.execute("SET hnsw.iterative_scan = strict_order")
        ddl = Path(__file__).with_name("schema.sql").read_text().replace("{dim}", str(embedder.dim))
        self.conn.execute(ddl)

    # -- tasks and templates -------------------------------------------------

    def add_task(self, task: Task) -> None:
        self.conn.execute(
            "INSERT INTO task VALUES (%s, %s, %s, %s) ON CONFLICT (task_id) DO NOTHING",
            (task.task_id, task.task_family, task.template_desc, task.task_desc))
        known = self.conn.execute(
            "SELECT 1 FROM template WHERE task_family = %s AND template_desc = %s",
            (task.task_family, task.template_desc)).fetchone()
        if not known:
            self.conn.execute(
                "INSERT INTO template VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                (task.task_family, task.template_desc, self.embedder.encode([task.template_desc])[0]))

    # -- Maintain ------------------------------------------------------------

    def maintain(self, trace: Trace, k: int) -> bool:
        part = (trace.task_family, trace.template_desc, trace.quality_category)
        with self.conn.transaction():
            cur = self.conn.cursor()
            # Serialize concurrent writers of the same partition.
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", ("\x1f".join(part),))
            # The k-th ranked trace is the one τ has to beat once the partition is full.
            kth = cur.execute(
                f"SELECT quality_score, horizon FROM trace WHERE task_family = %s AND template_desc = %s "
                f"AND quality_category = %s ORDER BY {_ORDER} OFFSET %s LIMIT 1", (*part, k - 1)).fetchone()
            if kth and (-trace.quality_score, trace.horizon) >= (-kth["quality_score"], kth["horizon"]):
                return False
            cur.execute(
                f"INSERT INTO trace ({_TRACE_COLS}) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (trace.trace_id, trace.task_id, trace.task_family, trace.template_desc, trace.actor,
                 trace.label, trace.reward, trace.horizon, trace.quality_category, trace.quality_score))
            if trace.steps:
                cur.executemany("INSERT INTO step VALUES (%s, %s, %s, %s)",
                                [(trace.trace_id, s.step_index, s.observation, s.action) for s in trace.steps])
            cur.execute(
                f"DELETE FROM trace WHERE trace_id IN (SELECT trace_id FROM trace WHERE task_family = %s "
                f"AND template_desc = %s AND quality_category = %s ORDER BY {_ORDER} OFFSET %s)", (*part, k))
            cur.execute("DELETE FROM experience WHERE task_family = %s AND template_desc = %s", part[:2])
        return True

    # -- Retrieve ------------------------------------------------------------

    def retrieve(self, task, category, n, *, exclude_task_id=None, mode=EXACT_FALLBACK, query_vec=None,
                 with_steps=True):
        loo = "AND task_id <> %(exclude)s" if exclude_task_id is not None else ""
        params = {"family": task.task_family, "template": task.template_desc, "category": category,
                  "exclude": exclude_task_id, "n": n}
        rows = []
        if mode == EXACT_FALLBACK:
            # Equality lookup on the indexed partition key.
            rows = self.conn.execute(
                f"SELECT {_TRACE_COLS} FROM trace WHERE task_family = %(family)s AND template_desc = "
                f"%(template)s AND quality_category = %(category)s {loo} ORDER BY {_ORDER} LIMIT %(n)s",
                params).fetchall()
        if len(rows) < n:
            params["q"] = np.asarray(self.embed_query(task) if query_vec is None else query_vec, dtype=np.float32)
            params["m"] = self.fallback_templates
            params["rest"] = n - len(rows)
            own = "AND template_desc <> %(template)s" if mode == EXACT_FALLBACK else ""
            # Rank templates of the family by cosine distance, then take the
            # best traces of each candidate template in that order.
            rows += self.conn.execute(
                f"WITH cand AS MATERIALIZED (SELECT template_desc, embedding <=> %(q)s AS dist FROM template "
                f"WHERE task_family = %(family)s {own} ORDER BY embedding <=> %(q)s LIMIT %(m)s) "
                f"SELECT t.* FROM cand CROSS JOIN LATERAL (SELECT {_TRACE_COLS}, seq FROM trace "
                f"WHERE task_family = %(family)s AND template_desc = cand.template_desc "
                f"AND quality_category = %(category)s {loo} ORDER BY {_ORDER} LIMIT %(rest)s) t "
                f"ORDER BY cand.dist, t.quality_score DESC, t.horizon, t.seq LIMIT %(rest)s",
                params).fetchall()
        return self._hydrate(rows, with_steps)

    def _hydrate(self, rows, with_steps) -> list[Trace]:
        if not rows:
            return []
        ids = [r["trace_id"] for r in rows]
        descs = {r["task_id"]: r["task_desc"] for r in self.conn.execute(
            "SELECT task_id, task_desc FROM task WHERE task_id = ANY(%s)", ([r["task_id"] for r in rows],))}
        steps: dict = {i: [] for i in ids}
        if with_steps:
            for s in self.conn.execute(
                    "SELECT trace_id, step_index, observation, action FROM step WHERE trace_id = ANY(%s) "
                    "ORDER BY trace_id, step_index", (ids,)):
                steps[s["trace_id"]].append(Step(s["step_index"], s["observation"], s["action"]))
        out = []
        for r in rows:
            r = {c: r[c] for c in r if c != "seq"}
            out.append(Trace(**{**r, "trace_id": str(r["trace_id"])}, steps=steps[r["trace_id"]],
                             task_desc=descs.get(r["task_id"])))
        return out

    # -- experience cache ----------------------------------------------------

    def cache_lookup(self, task, strategy, synthesizer):
        r = self.conn.execute(
            'SELECT task_family, template_desc, strategy, synthesizer, content, trace_ids, '
            'extract(epoch FROM "timestamp") AS ts FROM experience WHERE task_family = %s '
            'AND template_desc = %s AND strategy = %s AND synthesizer = %s',
            (task.task_family, task.template_desc, strategy, synthesizer)).fetchone()
        if not r:
            return None
        return Experience(r["task_family"], r["template_desc"], r["strategy"], r["synthesizer"], r["content"],
                          [str(i) for i in r["trace_ids"]], float(r["ts"]))

    def cache_put(self, experience):
        e = experience
        self.conn.execute(
            'INSERT INTO experience VALUES (%s, %s, %s, %s, %s, %s::uuid[], to_timestamp(%s)) '
            'ON CONFLICT (task_family, template_desc, strategy, synthesizer) DO UPDATE SET '
            'content = EXCLUDED.content, trace_ids = EXCLUDED.trace_ids, "timestamp" = EXCLUDED."timestamp"',
            (e.task_family, e.template_desc, e.strategy, e.synthesizer, e.content, e.trace_ids, e.timestamp))

    def has_task_traces(self, task_id):
        return self.conn.execute("SELECT 1 FROM trace WHERE task_id = %s LIMIT 1", (task_id,)).fetchone() is not None

    def count_traces(self):
        return self.conn.execute("SELECT count(*) AS c FROM trace").fetchone()["c"]

    def close(self):
        self.conn.close()
