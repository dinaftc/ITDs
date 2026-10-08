"""Synthesis prompts and injection wrappers (Appendix A)."""

RULES_SYNTHESIS = """ROLE
You are a highly skilled expert in agent behaviour and behavioral pattern extraction. Your task is to extract conditional action patterns and to state what to do and what not to do in specific situations.

CONTEXT
{context}

TRAJECTORIES
Grouped by how far each one got: completed the task, got part of the way, or made no progress.
{experiences}

INSTRUCTIONS
In a <think> block, compare the trajectories: which actions were taken and in what order, which changed the environment state and which were rejected, and where the less successful ones diverged from the more successful ones. Then generalize into conditional rules of the form "when [situation], do [action sequence]" or "when [situation], never do [action]".

CONSTRAINTS
3-5 rules of each kind, 1-2 sentences each, specific enough to follow step by step. Use the action vocabulary from CONTEXT. Replace whatever varies between task instances with placeholders."""

REFLECTIONS_SYNTHESIS = """ROLE
You are a highly skilled expert in agent behaviour and behavioral analysis. Your task is to analyze why trajectories reached the outcomes they did and to explain the underlying reasons.

CONTEXT
{context}

TRAJECTORIES
Grouped by how far each one got.
{experiences}

INSTRUCTIONS
In a <think> block, identify what separates the trajectories: which action sequence carried the task furthest, how the environment responded at each step, and what stopped the less successful ones.

CONSTRAINTS
Every sentence references a concrete action or entity. Use the action vocabulary from CONTEXT; replace varying elements with placeholders."""

RULES_WRAPPER = """BEHAVIORAL RULES FROM PAST EXPERIENCES
{content}
Apply these rules to your current decision."""

REFLECTIONS_WRAPPER = """REFLECTION FROM PAST EXPERIENCES
{content}
Use this reflection to guide your behavior."""

EXAMPLES_WRAPPER = """PAST EXPERIENCES from prior tasks in this task family:
Treat these as demonstrations: follow the same approach as the successful ones, and avoid the approach taken in the failed ones.
{content}"""

# Headings under which traces are grouped, by quality category.
GROUP_TITLES = {
    "success": "completed the task",
    "near-success": "got part of the way",
    "failure": "made no progress",
}
