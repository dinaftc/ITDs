"""Command line: `itd run`, `itd summarize`, `itd bench`."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import yaml

from .llm import load_dotenv
from .runner import RunConfig, run, summarize


def _add_config_flags(parser: argparse.ArgumentParser) -> None:
    for f in dataclasses.fields(RunConfig):
        if f.name == "env":
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type == "bool":
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(flag, type={"int": int, "float": float}.get(f.type, str), default=None)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="itd")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run one configuration (flags override the YAML config)")
    p_run.add_argument("--config", help="YAML file with RunConfig fields")
    _add_config_flags(p_run)

    p_sum = sub.add_parser("summarize", help="success rates of finished runs")
    p_sum.add_argument("runs", nargs="+", type=Path)
    p_sum.add_argument("--by", help="group by a result field, e.g. task_family or interaction_type")

    p_bench = sub.add_parser("bench", help="operator latency (Section 4.8)")
    from . import bench
    bench.add_arguments(p_bench)

    args = parser.parse_args(argv)
    load_dotenv()
    if args.cmd == "run":
        values = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
        values.update({k: v for k, v in vars(args).items() if k not in ("cmd", "config") and v is not None})
        values.setdefault("dsn", os.environ.get("ITD_PG_DSN", ""))
        values.setdefault("qdrant_url", os.environ.get("ITD_QDRANT_URL", ""))
        run(RunConfig(**values))
    elif args.cmd == "summarize":
        for row in summarize(args.runs, args.by):
            print(json.dumps(row))
    elif args.cmd == "bench":
        bench.main(args)


if __name__ == "__main__":
    main()
