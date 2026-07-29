from __future__ import annotations

import hashlib
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from qa_engine import parse_decision, validate_rules


def _suggestion_id(parts: Iterable[str]) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_pattern_suggestions(
    reports: Mapping[str, Mapping[str, Any]],
    rules: Mapping[str, Any],
) -> list[dict[str, Any]]:
    normalized_rules = validate_rules(rules)
    protected = set(normalized_rules["protected_spelling_terms"])
    protected_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exact_evidence: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )

    for report_id, report in reports.items():
        reviews = report.get("reviews", {})
        for row in report.get("rows", []):
            review = reviews.get(str(row["issue_id"]), {})
            status, note = parse_decision(
                str(review.get("decision") or ""),
                str(review.get("review_note") or ""),
            )
            if not status:
                continue
            evidence = {
                "report_id": report_id,
                "report_name": report.get("name", report_id),
                "issue_id": row["issue_id"],
            }
            original = str(row.get("original") or "")
            proposed = str(row.get("proposed") or "")
            if (
                row.get("checker") == "check_spelling"
                and status == "rejected"
                and original.strip().lower() not in protected
            ):
                protected_evidence[original.strip().lower()].append(evidence)
                continue
            key = (
                str(row.get("checker") or ""),
                str(row.get("field") or ""),
                original,
                proposed,
                status,
                note or "",
            )
            exact_evidence[key].append(evidence)

    suggestions: list[dict[str, Any]] = []
    for term, evidence in protected_evidence.items():
        suggestions.append(
            {
                "selected": False,
                "suggestion_id": _suggestion_id(["protected_term", term]),
                "suggestion_type": "protected spelling term",
                "checker": "check_spelling",
                "field": "*",
                "original": term,
                "proposed": "*",
                "status": "rejected",
                "comment": "Human reviewers confirmed this is a valid term.",
                "evidence_count": len(evidence),
                "reports": ", ".join(sorted({item["report_name"] for item in evidence})),
            }
        )
    for key, evidence in exact_evidence.items():
        checker, field, original, proposed, status, comment = key
        suggestions.append(
            {
                "selected": False,
                "suggestion_id": _suggestion_id(["exact", *key]),
                "suggestion_type": "exact rule",
                "checker": checker,
                "field": field,
                "original": original,
                "proposed": proposed,
                "status": status,
                "comment": comment,
                "evidence_count": len(evidence),
                "reports": ", ".join(sorted({item["report_name"] for item in evidence})),
            }
        )
    return sorted(
        suggestions,
        key=lambda item: (
            -int(item["evidence_count"]),
            str(item["suggestion_type"]),
            str(item["original"]).lower(),
        ),
    )


def build_shared_pattern_suggestions(
    evidence_rows: Iterable[Mapping[str, Any]],
    rules: Mapping[str, Any],
) -> list[dict[str, Any]]:
    normalized_rules = validate_rules(rules)
    protected = set(normalized_rules["protected_spelling_terms"])
    protected_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exact_evidence: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )

    for source in evidence_rows:
        item = dict(source)
        original = str(item.get("original") or "")
        proposed = str(item.get("proposed") or "")
        status = str(item.get("status") or "")
        if (
            item.get("checker") == "check_spelling"
            and status == "rejected"
            and original.strip().lower() not in protected
        ):
            protected_evidence[original.strip().lower()].append(item)
            continue
        key = (
            str(item.get("checker") or ""),
            str(item.get("field") or ""),
            original,
            proposed,
            status,
            str(item.get("comment") or ""),
        )
        exact_evidence[key].append(item)

    def evidence_totals(items: list[dict[str, Any]]) -> dict[str, Any]:
        human_decisions = sum(int(item.get("human_decisions") or 0) for item in items)
        human_reports = max(
            (int(item.get("human_reports") or 0) for item in items),
            default=0,
        )
        occurrences = sum(int(item.get("occurrences") or 0) for item in items)
        course_coverage = max(
            (int(item.get("course_coverage") or 0) for item in items),
            default=0,
        )
        source_names = {
            name.strip()
            for item in items
            for field in ("reviewed_sources", "seen_in_reports")
            for name in str(item.get(field) or "").split(",")
            if name.strip()
        }
        return {
            "evidence_count": human_decisions,
            "human_reports": human_reports,
            "course_coverage": course_coverage,
            "occurrences": occurrences,
            "reports": ", ".join(sorted(source_names)),
        }

    suggestions: list[dict[str, Any]] = []
    for term, items in protected_evidence.items():
        suggestions.append(
            {
                "selected": False,
                "suggestion_id": _suggestion_id(["protected_term", term]),
                "suggestion_type": "protected spelling term",
                "checker": "check_spelling",
                "field": "*",
                "original": term,
                "proposed": "*",
                "status": "rejected",
                "comment": "Human reviewers confirmed this is a valid term.",
                **evidence_totals(items),
            }
        )
    for key, items in exact_evidence.items():
        checker, field, original, proposed, status, comment = key
        suggestions.append(
            {
                "selected": False,
                "suggestion_id": _suggestion_id(["exact", *key]),
                "suggestion_type": "exact rule",
                "checker": checker,
                "field": field,
                "original": original,
                "proposed": proposed,
                "status": status,
                "comment": comment,
                **evidence_totals(items),
            }
        )
    return sorted(
        suggestions,
        key=lambda item: (
            -int(item["course_coverage"]),
            -int(item["evidence_count"]),
            -int(item["occurrences"]),
            str(item["suggestion_type"]),
            str(item["original"]).lower(),
        ),
    )


def promote_suggestions(
    rules: Mapping[str, Any], suggestions: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    updated = validate_rules(deepcopy(dict(rules)))
    protected = set(updated["protected_spelling_terms"])
    exact_rules = list(updated["exact_rules"])
    exact_signatures = {
        (
            rule.get("checker", "*"),
            rule.get("field", "*"),
            rule.get("original", "*"),
            rule.get("proposed", "*"),
            rule.get("status"),
            rule.get("comment", ""),
        )
        for rule in exact_rules
    }

    for suggestion in suggestions:
        if not suggestion.get("selected"):
            continue
        if suggestion.get("suggestion_type") == "protected spelling term":
            protected.add(str(suggestion.get("original") or "").strip().lower())
            continue
        signature = (
            suggestion.get("checker", "*"),
            suggestion.get("field", "*"),
            suggestion.get("original", "*"),
            suggestion.get("proposed", "*"),
            suggestion.get("status"),
            suggestion.get("comment", ""),
        )
        if signature in exact_signatures:
            continue
        exact_rules.append(
            {
                "checker": signature[0],
                "field": signature[1],
                "original": signature[2],
                "proposed": signature[3],
                "status": signature[4],
                "comment": signature[5],
                "source": "human_feedback",
            }
        )
        exact_signatures.add(signature)

    updated["protected_spelling_terms"] = sorted(protected)
    updated["exact_rules"] = exact_rules
    updated["rules_version"] = datetime.now(timezone.utc).strftime("%Y.%m.%d.%H%M%S")
    return validate_rules(updated)
