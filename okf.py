"""Minimal OKF bundle reader: a folder of Markdown files with YAML frontmatter."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
_FM = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.S)
_SKIP = {"index.md", "log.md"}

_docs: dict[str, dict[str, Any]] | None = None


def load_bundle(root: Path = KNOWLEDGE_DIR) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.md")):
        if path.name in _SKIP:
            continue
        m = _FM.match(path.read_text(encoding="utf-8"))
        if not m:
            continue
        meta = yaml.safe_load(m.group(1)) or {}
        if not meta.get("type"):  # OKF's one hard requirement
            continue
        docs[path.stem] = {"id": path.stem, "meta": meta, "body": m.group(2).strip()}
    return docs


def get_docs() -> dict[str, dict[str, Any]]:
    global _docs
    if _docs is None:
        _docs = load_bundle()
    return _docs


def get(topic: str) -> dict[str, Any] | None:
    return get_docs().get(topic)


def search(query: str, k: int = 2) -> list[dict[str, Any]]:
    words = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 2}
    scored = []
    for d in get_docs().values():
        meta = d["meta"]
        tags = " ".join(map(str, meta.get("tags") or [])).lower()
        head = f"{meta.get('title', '')} {meta.get('description', '')}".lower()
        body = d["body"].lower()
        score = sum(3 * tags.count(w) + 2 * head.count(w) + body.count(w) for w in words)
        if score:
            scored.append((score, d))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:k]]