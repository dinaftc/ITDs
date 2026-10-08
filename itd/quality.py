"""Quality categories and scores (Section 3.1, "Quality category and partition")."""
from __future__ import annotations

from .model import FAILURE, NEAR_SUCCESS, SUCCESS, Trace


def assign_quality(trace: Trace, theta: float = 0.5) -> Trace:
    """Set quality_category and quality_score in place.

    success:       q = reward / horizon          (favors efficient execution)
    near-success:  q = m / M, when m / M >= theta (favors greater progress)
    failure:       q = 1 - m / M                  (favors earlier failure)
    """
    if trace.label == SUCCESS:
        trace.quality_category = SUCCESS
        trace.quality_score = trace.reward / max(trace.horizon, 1)
        return trace
    if trace.progress is None:
        raise ValueError(f"unsuccessful trace {trace.trace_id} has no progress (m/M)")
    q_near = min(max(trace.progress, 0.0), 1.0)
    if q_near >= theta:
        trace.quality_category = NEAR_SUCCESS
        trace.quality_score = q_near
    else:
        trace.quality_category = FAILURE
        trace.quality_score = 1.0 - q_near
    return trace


def rank_key(trace: Trace) -> tuple[float, int]:
    """Sort key for ranking within a partition: higher score, then shorter horizon."""
    return (-trace.quality_score, trace.horizon)


def outranks(a: Trace, b: Trace) -> bool:
    return rank_key(a) < rank_key(b)
