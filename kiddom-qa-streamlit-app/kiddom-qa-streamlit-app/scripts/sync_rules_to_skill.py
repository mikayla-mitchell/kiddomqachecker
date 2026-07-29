#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def validate_rules(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        rules = json.load(handle)
    if rules.get("schema_version") != 1:
        raise ValueError("Expected schema_version 1.")
    if not isinstance(rules.get("protected_spelling_terms"), list):
        raise ValueError("protected_spelling_terms must be a list.")
    if not isinstance(rules.get("safe_typo_targets"), list):
        raise ValueError("safe_typo_targets must be a list.")
    if not isinstance(rules.get("exact_rules"), list):
        raise ValueError("exact_rules must be a list.")
    return rules


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install an exported app rulebook as an overlay for the Codex skill."
    )
    parser.add_argument("rules_json", type=Path)
    parser.add_argument(
        "--skill-dir",
        type=Path,
        default=Path.home() / ".codex" / "skills" / "kiddom-qa-report-review",
    )
    args = parser.parse_args()

    source = args.rules_json.expanduser().resolve()
    skill_dir = args.skill_dir.expanduser().resolve()
    destination = skill_dir / "references" / "app_rules.json"
    classifier = skill_dir / "scripts" / "classify.py"

    validate_rules(source)
    if not classifier.exists():
        raise FileNotFoundError(
            f"No kiddom-qa-report-review skill found at {skill_dir}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    print(f"Synced rules to {destination}")


if __name__ == "__main__":
    main()
