"""Policy-aware reusable workflow playbooks for agents."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Playbook:
    id: str
    title: str
    summary: str
    triggers: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    steps: tuple[str, ...]
    source: str = "local"
    version: str = "1"

    def catalog_entry(self) -> dict[str, Any]:
        """Small index entry safe to inject into context."""
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "triggers": list(self.triggers),
            "allowed_tools": list(self.allowed_tools),
            "source": self.source,
            "version": self.version,
        }

    def full_body(self) -> dict[str, Any]:
        body = self.catalog_entry()
        body["steps"] = list(self.steps)
        return body


def load_playbooks(directory: str | Path) -> list[Playbook]:
    """Load playbooks from YAML files in a directory."""
    root = Path(directory)
    if not root.exists():
        return []

    playbooks: list[Playbook] = []
    for path in sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for raw in data.get("playbooks", []):
            playbooks.append(
                Playbook(
                    id=str(raw["id"]),
                    title=str(raw["title"]),
                    summary=str(raw.get("summary", "")),
                    triggers=tuple(str(v) for v in raw.get("triggers", [])),
                    allowed_tools=tuple(str(v) for v in raw.get("allowed_tools", [])),
                    steps=tuple(str(v) for v in raw.get("steps", [])),
                    source=str(raw.get("source", path.name)),
                    version=str(raw.get("version", "1")),
                )
            )
    return playbooks


def catalog(playbooks: list[Playbook], limit: int = 20) -> list[dict[str, Any]]:
    """Return compact playbook entries without step bodies."""
    return [playbook.catalog_entry() for playbook in playbooks[:limit]]


def recommend_playbooks(
    query: str,
    playbooks: list[Playbook],
    allowed_tools: set[str] | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Rank playbooks by lexical trigger overlap and policy compatibility."""
    query_terms = _terms(query)
    ranked: list[tuple[float, Playbook]] = []

    for playbook in playbooks:
        required = set(playbook.allowed_tools)
        if allowed_tools is not None and not required.issubset(allowed_tools):
            continue

        haystack = " ".join([playbook.title, playbook.summary, *playbook.triggers])
        terms = _terms(haystack)
        overlap = len(query_terms & terms)
        exact_trigger_bonus = sum(1 for trigger in playbook.triggers if trigger.lower() in query.lower())
        score = overlap + exact_trigger_bonus * 3
        if score > 0:
            ranked.append((float(score), playbook))

    ranked.sort(key=lambda item: (-item[0], item[1].id))
    return [
        {"score": score, **playbook.catalog_entry()}
        for score, playbook in ranked[:limit]
    ]


def get_playbook(playbooks: list[Playbook], playbook_id: str) -> Playbook | None:
    for playbook in playbooks:
        if playbook.id == playbook_id:
            return playbook
    return None


def mask_env_values(text: str, env_names: list[str]) -> str:
    """Replace configured secret values with ${NAME} placeholders."""
    masked = text
    for name in env_names:
        value = os.environ.get(name, "")
        if value:
            masked = masked.replace(value, "${" + name + "}")
    return masked


def propose_playbook_from_trace(
    playbook_id: str,
    title: str,
    summary: str,
    steps: list[str],
    allowed_tools: list[str],
    triggers: list[str] | None = None,
) -> dict[str, Any]:
    """Build a playbook dict from a repeated successful workflow trace."""
    normalized_id = re.sub(r"[^a-z0-9_]+", "_", playbook_id.lower()).strip("_")
    return {
        "id": normalized_id,
        "title": title.strip(),
        "summary": summary.strip(),
        "triggers": triggers or _fallback_triggers(title, summary),
        "allowed_tools": sorted(set(allowed_tools)),
        "steps": [step.strip() for step in steps if step.strip()],
        "source": "proposed",
        "version": "1",
    }


def _fallback_triggers(title: str, summary: str) -> list[str]:
    terms = sorted(_terms(f"{title} {summary}"))
    return terms[:8]


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9_]{3,}", text.lower())
        if term not in {"the", "and", "for", "with", "that", "this", "from"}
    }
