# Example data

Records from real runs.

| File | Content |
|---|---|
| `alfworld_trace.json` | an ALFWorld task, a successful trace, its seven steps |
| `alfworld_experiences.json` | the same two traces under each reuse strategy |
| `webarena_trace_and_experience.json` | a GitLab task, a four-step trace, its Rules experience |

A trace carries the fields the store ranks on:

```json
"label": "success", "reward": 1.0, "horizon": 7,
"quality_category": "success", "quality_score": 0.142857
```

An Experience names the `trace_ids` it was built from. `synthesizer` is empty
for Examples, which concatenates the traces directly; Rules and Reflections
record the model that wrote them.
