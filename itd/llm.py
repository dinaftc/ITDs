"""Chat-completion clients for task solvers and synthesizers.

Models are declared in configs/models.yaml and served through any
OpenAI-compatible endpoint (DeepSeek API, vLLM, ...).
"""
from __future__ import annotations

import os
import random
import re
import time
from pathlib import Path

import yaml


class LLM:
    def __init__(self, name: str, model: str, base_url: str, api_key: str = "EMPTY",
                 temperature: float = 0.1, max_tokens: int = 1024, retries: int = 5):
        from openai import OpenAI

        self.name, self.model = name, model
        self.temperature, self.max_tokens, self.retries = temperature, max_tokens, retries
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        self.seconds = 0.0   # cumulated wall-clock time spent in calls

    def chat(self, messages: list[dict]) -> str:
        start = time.perf_counter()
        try:
            for attempt in range(self.retries):
                try:
                    out = self.client.chat.completions.create(
                        model=self.model, messages=messages, temperature=self.temperature,
                        max_tokens=self.max_tokens)
                    return out.choices[0].message.content or ""
                except Exception:
                    if attempt == self.retries - 1:
                        raise
                    time.sleep(2 ** attempt)
        finally:
            self.seconds += time.perf_counter() - start


class MockLLM:
    """Offline stand-in used by the dry-run configuration and the tests: as a
    task solver it picks a random admissible action; as a synthesizer it
    returns a placeholder. It says nothing about experience reuse."""

    def __init__(self, name: str = "mock", seed: int = 0):
        self.name, self.seconds = name, 0.0
        self.rng = random.Random(seed)

    def chat(self, messages: list[dict]) -> str:
        last = messages[-1]["content"]
        m = re.search(r"Admissible actions: (.*)", last)
        if m:
            return "Thought: try something.\nAction: " + self.rng.choice(m.group(1).split("; "))
        return "<think>comparing trajectories</think>\n(mock synthesis)"


def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def make_llm(name: str, models_file: str = "configs/models.yaml", seed: int = 0):
    if name == "mock":
        return MockLLM(seed=seed)
    load_dotenv()
    spec = yaml.safe_load(Path(models_file).read_text())[name]
    spec = {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in spec.items()}
    return LLM(name, **spec)
