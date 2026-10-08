"""Qdrant instantiation (Sections 3.1 and 4.8).

Traces and task templates live in separate collections. Each trace point
stores, as payload, the attributes used for partition filtering and trace
ranking (payload indices); each template point stores its description and
embedding (HNSW), so similarity retrieval ranks templates before traces are
selected. Collections are prefixed by the ITD namespace.
"""
from __future__ import annotations

import time
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from ..model import Experience, Step, Task, Trace
from .base import EXACT_FALLBACK, ITDStore

_NS = uuid.UUID("6f1d2c1e-6a51-4d0b-9d3e-1f0c5a7e9b10")
# Qdrant orders a scroll by a single payload key. `rank` folds the horizon
# tie-break into the quality score; scores are ratios of small integers, so a
# 1e-9-per-step penalty never reorders traces whose scores differ.
_EPS = 1e-9


def _tid(family: str, template: str) -> str:
    return str(uuid.uuid5(_NS, family + "\x1f" + template))


def _rank(trace: Trace) -> float:
    return trace.quality_score - _EPS * trace.horizon


def _match(key, value):
    return qm.FieldCondition(key=key, match=qm.MatchValue(value=value))


class QdrantStore(ITDStore):
    def __init__(self, embedder, url: str | None = None, namespace: str = "itd", *, reset: bool = False,
                 fallback_templates: int = 20, client: QdrantClient | None = None):
        super().__init__(embedder)
        self.client = client or (QdrantClient(url=url) if url else QdrantClient(":memory:"))
        self.fallback_templates = fallback_templates
        self.traces, self.templates, self.experiences = (f"{namespace}_{c}" for c in ("traces", "templates", "experiences"))
        self._task_descs: dict[str, str] = {}
        if reset:
            for c in (self.traces, self.templates, self.experiences):
                self.client.delete_collection(c)
        if not self.client.collection_exists(self.templates):
            self.client.create_collection(
                self.templates, vectors_config=qm.VectorParams(size=embedder.dim, distance=qm.Distance.COSINE))
            self.client.create_payload_index(self.templates, "task_family", qm.PayloadSchemaType.KEYWORD)
            # Traces and experiences are only ever filtered, never searched by vector.
            for c in (self.traces, self.experiences):
                self.client.create_collection(c, vectors_config={})
            for key in ("template_id", "quality_category", "task_id"):
                self.client.create_payload_index(self.traces, key, qm.PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(self.traces, "rank", qm.PayloadSchemaType.FLOAT)
            self.client.create_payload_index(self.experiences, "template_id", qm.PayloadSchemaType.KEYWORD)

    # -- tasks and templates -------------------------------------------------

    def add_task(self, task: Task) -> None:
        tid = _tid(task.task_family, task.template_desc)
        if not self.client.retrieve(self.templates, [tid], with_payload=False):
            self.client.upsert(self.templates, [qm.PointStruct(
                id=tid, vector=self.embedder.encode([task.template_desc])[0].tolist(),
                payload={"task_family": task.task_family, "template_desc": task.template_desc})])
        self._task_descs[task.task_id] = task.task_desc

    # -- Maintain ------------------------------------------------------------

    def trace_point(self, trace: Trace) -> qm.PointStruct:
        payload = {key: v for key, v in trace.to_dict().items() if key not in ("progress", "trace_id")}
        # task_desc is denormalized into the trace payload (no join at retrieval time).
        payload.update(template_id=_tid(trace.task_family, trace.template_desc), rank=_rank(trace),
                       seq=time.time_ns(), task_desc=trace.task_desc or self._task_descs[trace.task_id])
        return qm.PointStruct(id=trace.trace_id, vector={}, payload=payload)

    def _partition_filter(self, tid, category, exclude_task_id=None):
        must_not = [_match("task_id", exclude_task_id)] if exclude_task_id is not None else None
        return qm.Filter(must=[_match("template_id", tid), _match("quality_category", category)], must_not=must_not)

    def _scroll(self, flt, limit, with_payload=True):
        points, _ = self.client.scroll(
            self.traces, scroll_filter=flt, limit=limit, with_payload=with_payload, with_vectors=False,
            order_by=qm.OrderBy(key="rank", direction=qm.Direction.DESC))
        return points

    def maintain(self, trace: Trace, k: int) -> bool:
        tid = _tid(trace.task_family, trace.template_desc)
        rank = _rank(trace)
        # The k best traces of the partition; τ must beat the last one once it is full.
        top = self._scroll(self._partition_filter(tid, trace.quality_category), k, with_payload=["rank", "seq"])
        top.sort(key=lambda p: (-p.payload["rank"], p.payload["seq"]))
        if len(top) >= k and rank <= top[-1].payload["rank"]:
            return False
        self.client.upsert(self.traces, [self.trace_point(trace)])
        if len(top) >= k:
            self.client.delete(self.traces, qm.PointIdsList(points=[top[-1].id]))
        self.client.delete(self.experiences, qm.FilterSelector(filter=qm.Filter(must=[_match("template_id", tid)])))
        return True

    # -- Retrieve ------------------------------------------------------------

    def retrieve(self, task, category, n, *, exclude_task_id=None, mode=EXACT_FALLBACK, query_vec=None,
                 with_steps=True):
        own = _tid(task.task_family, task.template_desc)
        payload = True if with_steps else qm.PayloadSelectorExclude(exclude=["steps"])
        points = []
        if mode == EXACT_FALLBACK:
            points = self._scroll(self._partition_filter(own, category, exclude_task_id), n, payload)
        if len(points) < n:
            q = self.embed_query(task) if query_vec is None else query_vec
            flt = qm.Filter(must=[_match("task_family", task.task_family)],
                            must_not=[qm.HasIdCondition(has_id=[own])] if mode == EXACT_FALLBACK else None)
            candidates = self.client.query_points(
                self.templates, query=np.asarray(q).tolist(), query_filter=flt, limit=self.fallback_templates,
                with_payload=False).points
            for cand in candidates:
                points += self._scroll(self._partition_filter(cand.id, category, exclude_task_id),
                                       n - len(points), payload)
                if len(points) >= n:
                    break
        return [self._to_trace(p) for p in points]

    @staticmethod
    def _to_trace(point) -> Trace:
        d = {k: v for k, v in point.payload.items() if k not in ("template_id", "rank", "seq")}
        d["steps"] = [Step(**s) for s in d.get("steps", [])]
        return Trace(trace_id=str(point.id), **d)

    # -- experience cache ----------------------------------------------------

    @staticmethod
    def _eid(family, template, strategy, synthesizer) -> str:
        return str(uuid.uuid5(_NS, "\x1f".join((family, template, strategy, synthesizer))))

    def cache_lookup(self, task, strategy, synthesizer):
        hit = self.client.retrieve(
            self.experiences, [self._eid(task.task_family, task.template_desc, strategy, synthesizer)])
        if not hit:
            return None
        return Experience(**{k: v for k, v in hit[0].payload.items() if k != "template_id"})

    def cache_put(self, experience):
        e = experience
        payload = {**e.__dict__, "template_id": _tid(e.task_family, e.template_desc)}
        self.client.upsert(self.experiences, [qm.PointStruct(
            id=self._eid(e.task_family, e.template_desc, e.strategy, e.synthesizer), vector={}, payload=payload)])

    def has_task_traces(self, task_id):
        return self.client.count(self.traces, count_filter=qm.Filter(must=[_match("task_id", task_id)])).count > 0

    def count_traces(self):
        return self.client.count(self.traces).count

    def close(self):
        self.client.close()
