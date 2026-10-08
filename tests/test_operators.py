"""Operator semantics (Section 3.3), checked on every available instantiation.

PostgreSQL is tested when ITD_TEST_PG_DSN points to a server with pgvector.
"""
import os
import uuid

import pytest

from itd.embeddings import HashEmbedder
from itd.model import CATEGORIES, FAILURE, NEAR_SUCCESS, SUCCESS, Experience, Step, Task, Trace
from itd.quality import assign_quality
from itd.store import SIMILARITY_ONLY, make_store

MAP = "Map"
TWO_ROUTES = "What is the duration required to walk from A to B, then drive from B to C?"
DRIVE = "How long does it take to drive from A to B?"
PHONE = "What is the phone number of A?"


@pytest.fixture(params=["memory", "qdrant", "postgres"])
def store(request):
    kw = {}
    if request.param == "qdrant":
        pytest.importorskip("qdrant_client")
    if request.param == "postgres":
        kw["dsn"] = os.environ.get("ITD_TEST_PG_DSN")
        if not kw["dsn"]:
            pytest.skip("ITD_TEST_PG_DSN not set")
    s = make_store(request.param, HashEmbedder(), "itd_test_" + uuid.uuid4().hex[:8], reset=False, **kw)
    yield s
    s.close()


def task(i, template=TWO_ROUTES, family=MAP, desc=None):
    return Task(f"task-{i}", family, template, desc or f"{template} (instance {i})")


def trace(t: Task, *, success=False, horizon=10, progress=0.0, theta=0.5):
    tr = Trace(t.task_id, t.task_family, t.template_desc, "tester", SUCCESS if success else FAILURE,
               1.0 if success else 0.0, horizon, [Step(0, "obs", "act"), Step(1, "obs2", "act2")],
               progress=progress, task_desc=t.task_desc)
    return assign_quality(tr, theta)


def put(store, tr, t, k=1):
    store.add_task(t)
    return store.maintain(tr, k)


def test_quality_assignment():
    t = task(0)
    assert trace(t, success=True, horizon=13).quality_score == pytest.approx(1 / 13)
    near = trace(t, progress=3 / 5)
    assert (near.quality_category, near.quality_score) == (NEAR_SUCCESS, pytest.approx(0.6))
    fail = trace(t, progress=1 / 5)
    assert (fail.quality_category, fail.quality_score) == (FAILURE, pytest.approx(0.8))
    assert trace(t, progress=0.6, theta=0.7).quality_category == FAILURE


def test_maintain_keeps_top_k_per_partition(store):
    t = task(0)
    # failure partition, k = 1: no progress (1.0) replaces one-of-five subtasks (0.8)
    assert put(store, trace(t, progress=1 / 5), t)
    assert put(store, trace(t, progress=0.0), t)
    assert not put(store, trace(t, progress=2 / 5), t)
    # success partition: a 13-step trace replaces a 14-step trace, not the reverse
    assert put(store, trace(t, success=True, horizon=14), t)
    assert put(store, trace(t, success=True, horizon=13), t)
    assert not put(store, trace(t, success=True, horizon=14), t)
    # near-success is maintained separately and cannot displace the others
    assert put(store, trace(t, progress=3 / 5), t)
    assert store.count_traces() == 3
    got = {tr.quality_category: tr for tr in store.retrieve_all(t, 1)}
    assert set(got) == set(CATEGORIES)
    assert got[SUCCESS].horizon == 13 and got[FAILURE].quality_score == pytest.approx(1.0)
    assert [s.action for s in got[SUCCESS].steps] == ["act", "act2"]
    assert got[SUCCESS].task_desc == t.task_desc


def test_maintain_tie_breaks_on_shorter_horizon(store):
    t = task(0)
    assert put(store, trace(t, progress=0.6, horizon=20), t)
    assert not put(store, trace(t, progress=0.6, horizon=20), t)     # full tie: incumbent stays
    assert put(store, trace(t, progress=0.6, horizon=12), t)
    assert store.retrieve(t, NEAR_SUCCESS, 5)[0].horizon == 12
    assert store.count_traces() == 1


def test_retrieve_ranks_within_partition(store):
    t = task(0)
    for h in (14, 9, 20, 11):
        put(store, trace(t, success=True, horizon=h), t, k=3)
    assert [tr.horizon for tr in store.retrieve(t, SUCCESS, 3)] == [9, 11, 14]
    assert [tr.horizon for tr in store.retrieve(t, SUCCESS, 2)] == [9, 11]


def test_similarity_fallback_within_family(store):
    drive, phone, other = task(1, DRIVE), task(2, PHONE), task(3, DRIVE, family="GitLab")
    put(store, trace(drive, success=True, horizon=5), drive)
    put(store, trace(phone, success=True, horizon=7), phone)
    put(store, trace(other, success=True, horizon=3), other)
    # unseen Map template: falls back to the closest stored Map template, never to another family
    new = task(9, "How long does it take to drive from A to C via B?",
               desc="How long does it take to drive from CMU to the airport via downtown?")
    got = store.retrieve(new, SUCCESS, 2)
    assert [tr.template_desc for tr in got] == [DRIVE, PHONE]
    assert all(tr.task_family == MAP for tr in got)
    # exact template first, fallback only fills the remaining positions
    put(store, trace(new, success=True, horizon=30), new)
    assert [tr.template_desc for tr in store.retrieve(new, SUCCESS, 2)] == [new.template_desc, DRIVE]
    assert len(store.retrieve(new, SUCCESS, 1)) == 1
    assert store.retrieve(new, FAILURE, 1) == []
    # similarity-only ranks every template by similarity (the exact one is simply the closest here)
    assert store.retrieve(new, SUCCESS, 1, mode=SIMILARITY_ONLY)[0].template_desc == new.template_desc


def test_leave_one_out(store):
    a, b = task(1), task(2)
    put(store, trace(a, success=True, horizon=5), a, k=2)
    put(store, trace(b, success=True, horizon=8), b, k=2)
    assert [tr.task_id for tr in store.retrieve(a, SUCCESS, 2, exclude_task_id=a.task_id)] == [b.task_id]
    assert store.has_task_traces(a.task_id) and not store.has_task_traces("task-404")


def test_cache_is_invalidated_by_partition_updates(store):
    t, other = task(0), task(1, DRIVE)
    kept = trace(t, success=True, horizon=14)
    put(store, kept, t)
    for tk in (t, other):
        store.add_task(tk)
        store.cache_put(Experience(tk.task_family, tk.template_desc, "rules", "m", "DO: x", [kept.trace_id]))
    hit = store.cache_lookup(t, "rules", "m")
    assert hit.content == "DO: x" and hit.trace_ids == [kept.trace_id]
    assert store.cache_lookup(t, "reflections", "m") is None
    assert not put(store, trace(t, success=True, horizon=15), t)     # rejected: cache stays valid
    assert store.cache_lookup(t, "rules", "m") is not None
    assert put(store, trace(t, success=True, horizon=13), t)         # retained: cache invalidated
    assert store.cache_lookup(t, "rules", "m") is None
    assert store.cache_lookup(other, "rules", "m") is not None       # other templates are untouched
