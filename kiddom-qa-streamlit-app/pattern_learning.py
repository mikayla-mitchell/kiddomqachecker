from __future__ import annotations

import hashlib
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from qa_engine import parse_decision, validate_rules


AUTOMATIC_RULE_SOURCE = "automatic_pattern_learning"


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
                "comment": "Valid curriculum or technical term.",
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
                "comment": "Valid curriculum or technical term.",
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


def automatic_suggestion_reason(suggestion: Mapping[str, Any]) -> str | None:
    """Return why a consensus pattern is safe to promote without an editor."""
    status = str(suggestion.get("status") or "")
    if status not in {"approved", "rejected"}:
        return None

    suggestion_type = str(suggestion.get("suggestion_type") or "")
    checker = str(suggestion.get("checker") or "")
    comment = str(suggestion.get("comment") or "").casefold()
    human_decisions = int(suggestion.get("evidence_count") or 0)
    human_reports = int(suggestion.get("human_reports") or 0)
    course_coverage = int(suggestion.get("course_coverage") or 0)
    occurrences = int(suggestion.get("occurrences") or 0)

    if (
        suggestion_type == "protected spelling term"
        and status == "rejected"
        and human_decisions >= 1
        and course_coverage >= 2
        and occurrences >= 2
    ):
        return "Confirmed valid term recurring across courses."

    if suggestion_type != "exact rule":
        return None

    if checker == "check_math" and status == "rejected" and human_decisions >= 1:
        return "Confirmed math-checker false positive."

    if (
        checker == "check_spacing"
        and human_decisions >= 2
        and human_reports >= 2
        and course_coverage >= 2
    ):
        return "Consistent spacing decision across reviewed courses."

    if (
        checker == "check_capitalization"
        and status == "rejected"
        and human_decisions >= 2
        and human_reports >= 2
        and course_coverage >= 2
        and any(
            signal in comment
            for signal in ("math", "notation", "constant", "standard", "lowercase")
        )
    ):
        return "Consistent math or standards notation across reviewed courses."

    if (
        checker == "check_spelling"
        and human_decisions >= 2
        and human_reports >= 2
        and course_coverage >= 2
    ):
        return "Consistent spelling decision across reviewed courses."

    return None


def automatic_suggestions(
    suggestions: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select consensus patterns that meet the conservative automation policy."""
    selected: list[dict[str, Any]] = []
    for source in suggestions:
        reason = automatic_suggestion_reason(source)
        if not reason:
            continue
        suggestion = dict(source)
        suggestion["selected"] = True
        suggestion["source"] = AUTOMATIC_RULE_SOURCE
        suggestion["automation_reason"] = reason
        selected.append(suggestion)
    return selected


def _suggestion_is_covered(
    rules: Mapping[str, Any], suggestion: Mapping[str, Any]
) -> bool:
    if suggestion.get("suggestion_type") == "protected spelling term":
        term = str(suggestion.get("original") or "").strip().casefold()
        return term in set(rules.get("protected_spelling_terms", []))
    signature = (
        suggestion.get("checker", "*"),
        suggestion.get("field", "*"),
        suggestion.get("original", "*"),
        suggestion.get("proposed", "*"),
        suggestion.get("status"),
        suggestion.get("comment", ""),
    )
    return signature in {
        (
            rule.get("checker", "*"),
            rule.get("field", "*"),
            rule.get("original", "*"),
            rule.get("proposed", "*"),
            rule.get("status"),
            rule.get("comment", ""),
        )
        for rule in rules.get("exact_rules", [])
    }


def promote_automatic_suggestions(
    rules: Mapping[str, Any],
    suggestions: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Promote newly qualified patterns and return the rules that were added."""
    normalized_rules = validate_rules(rules)
    promoted = [
        suggestion
        for suggestion in automatic_suggestions(suggestions)
        if not _suggestion_is_covered(normalized_rules, suggestion)
    ]
    if not promoted:
        return normalized_rules, []
    return promote_suggestions(normalized_rules, promoted), promoted


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
                "source": str(suggestion.get("source") or "human_feedback"),
            }
        )
        exact_signatures.add(signature)

    updated["protected_spelling_terms"] = sorted(protected)
    updated["exact_rules"] = exact_rules
    updated["rules_version"] = datetime.now(timezone.utc).strftime("%Y.%m.%d.%H%M%S")
    return validate_rules(updated)
