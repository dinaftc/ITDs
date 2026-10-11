"""Turn retrieved traces into an Experience and wrap it for the solver prompt.

Examples injects the trajectories themselves, Rules and Reflections ask a
synthesizer for do/don't directives or a diagnosis, Combined gives all three.
"""
from __future__ import annotations

import re

from . import prompts
from .model import CATEGORIES, COMBINED, EXAMPLES, REFLECTIONS, RULES, Experience, Task, Trace

_SYNTHESIS = {RULES: prompts.RULES_SYNTHESIS, REFLECTIONS: prompts.REFLECTIONS_SYNTHESIS}
_WRAPPERS = {EXAMPLES: prompts.EXAMPLES_WRAPPER, RULES: prompts.RULES_WRAPPER,
             REFLECTIONS: prompts.REFLECTIONS_WRAPPER}


def components(strategy: str) -> list[str]:
    """Combined presents Examples, Rules and Reflections derived from the same traces."""
    return [EXAMPLES, RULES, REFLECTIONS] if strategy == COMBINED else [strategy]


def render_traces(traces: list[Trace], max_obs_chars: int = 300) -> str:
    """Trajectories grouped by quality category, as expected by {experiences}."""
    blocks = []
    for category in CATEGORIES:
        group = [t for t in traces if t.quality_category == category]
        if not group:
            continue
        blocks.append(f"## Trajectories that {prompts.GROUP_TITLES[category]}")
        for i, t in enumerate(group, 1):
            lines = [f"### Trajectory {i} ({t.horizon} steps) -- task: {t.task_desc}"]
            for s in t.steps:
                obs = " ".join(s.observation.split())
                if len(obs) > max_obs_chars:
                    obs = obs[:max_obs_chars] + " ..."
                lines.append(f"[{s.step_index}] observation: {obs}\n[{s.step_index}] action: {s.action}")
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def materialize(traces: list[Trace], strategy: str, synthesizer, task: Task, context: str,
                max_obs_chars: int = 300) -> Experience:
    """materialize(T, s, σ) of Algorithm 1, for one non-combined strategy."""
    rendered = render_traces(traces, max_obs_chars)
    if strategy == EXAMPLES:
        content, synth_name = rendered, ""      # traces are presented directly
    else:
        prompt = _SYNTHESIS[strategy].format(context=context, experiences=rendered)
        reply = synthesizer.chat([{"role": "user", "content": prompt}])
        content = re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()
        synth_name = synthesizer.name
    return Experience(task.task_family, task.template_desc, strategy, synth_name, content,
                      [t.trace_id for t in traces])


def inject(experiences: list[Experience]) -> str:
    return "\n\n".join(_WRAPPERS[e.strategy].format(content=e.content) for e in experiences)
