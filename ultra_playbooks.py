#!/usr/bin/env python3
"""Inspect and recommend policy-aware Ultra RAG agent playbooks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent_playbooks import (
    catalog,
    get_playbook,
    load_playbooks,
    mask_env_values,
    propose_playbook_from_trace,
    recommend_playbooks,
)
from src.config import get_config


ROOT = Path(__file__).parent


def _settings() -> dict:
    cfg = get_config()
    return cfg.get("agent_playbooks", {})


def _playbooks_dir() -> Path:
    configured = _settings().get("dir", "playbooks")
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


def _allowed_tools() -> set[str]:
    return set(_settings().get("allowed_tools", []))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ultra RAG reusable agent playbooks")
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="List compact playbook catalog")
    list_cmd.add_argument("--json", action="store_true")

    show_cmd = sub.add_parser("show", help="Show full playbook body")
    show_cmd.add_argument("id")
    show_cmd.add_argument("--json", action="store_true")

    rec_cmd = sub.add_parser("recommend", help="Recommend playbooks for a query")
    rec_cmd.add_argument("query")
    rec_cmd.add_argument("--json", action="store_true")

    mask_cmd = sub.add_parser("mask", help="Mask configured environment secret values in text")
    mask_cmd.add_argument("text")

    propose_cmd = sub.add_parser("propose", help="Create a playbook draft from workflow steps")
    propose_cmd.add_argument("--id", required=True)
    propose_cmd.add_argument("--title", required=True)
    propose_cmd.add_argument("--summary", required=True)
    propose_cmd.add_argument("--tool", action="append", default=[])
    propose_cmd.add_argument("--step", action="append", default=[])

    args = parser.parse_args()
    playbooks = load_playbooks(_playbooks_dir())

    if args.command == "list":
        payload = catalog(playbooks, limit=int(_settings().get("max_catalog_items", 20)))
        _print(payload, args.json)
    elif args.command == "show":
        playbook = get_playbook(playbooks, args.id)
        if playbook is None:
            raise SystemExit(f"Unknown playbook: {args.id}")
        _print(playbook.full_body(), args.json)
    elif args.command == "recommend":
        payload = recommend_playbooks(args.query, playbooks, allowed_tools=_allowed_tools())
        _print(payload, args.json)
    elif args.command == "mask":
        print(mask_env_values(args.text, _settings().get("masked_env", [])))
    elif args.command == "propose":
        payload = propose_playbook_from_trace(
            args.id,
            args.title,
            args.summary,
            args.step,
            args.tool,
        )
        print(json.dumps(payload, indent=2))


def _print(payload, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
        return
    if isinstance(payload, list):
        for item in payload:
            score = f" score={item['score']}" if "score" in item else ""
            print(f"{item['id']}: {item['title']}{score}")
            print(f"  {item['summary']}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
