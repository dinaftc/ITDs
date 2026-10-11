# Example data

Records taken from real runs, so the shapes here are what the code actually
writes. Nothing was hand-written.

| File | Content |
|---|---|
| `alfworld_trace.json` | one ALFWorld task, the trace of a successful attempt, and its seven steps |
| `alfworld_experiences.json` | the same two traces turned into an Experience by each of the three strategies |
| `webarena_trace_and_experience.json` | a GitLab task, a four-step successful trace, and the Rules experience for its template |

## Reading a trace

A trace is one attempt at one task. The fields the store indexes on are the
quality category and the quality score:

```json
"label": "success", "reward": 1.0, "horizon": 7,
"quality_category": "success", "quality_score": 0.142857
```

`horizon` is the number of steps taken. For a successful trace the score is
`reward / horizon`, so a shorter solution scores higher and wins the partition
over a longer one. An unsuccessful trace is scored on how far it got instead:
`m/M` if that reaches θ (near-success), otherwise `1 - m/M` (failure), which
ranks the quick, clean failures above the ones that flailed.

Steps hold the observation the agent saw, the action it chose, and what came
back. That is what the Examples strategy replays and what the synthesizer
reads to write rules or a reflection.

## Reading an experience

An Experience is what retrieval actually hands the task solver, built from the
`trace_ids` it names. The three strategies differ only in how much is left to
the reader:

- **Examples** concatenates the traces themselves, labelled by outcome. No LLM
  is involved, so `synthesizer` is empty.
- **Rules** asks a synthesizer for do/don't rules abstracted over the traces.
- **Reflections** asks it for a short retrospective on what separated the
  successful attempt from the unsuccessful one.

All three are cached per (template, strategy, synthesizer) and invalidated
when `maintain` changes the partition they were built from, so the same
traces are never synthesized twice.
