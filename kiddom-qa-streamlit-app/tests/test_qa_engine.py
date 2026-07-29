from __future__ import annotations

import csv
import io

from comment_policy import training_comment
from pattern_learning import (
    build_pattern_suggestions,
    build_shared_pattern_suggestions,
    promote_suggestions,
)
from qa_engine import (
    apply_reviews,
    classify_records,
    final_csv_bytes,
    load_rules,
    parse_report_html,
    review_progress,
)


BASE_RULES = {
    "schema_version": 1,
    "rules_version": "test",
    "protected_spelling_terms": ["desmos"],
    "safe_typo_targets": ["the"],
    "exact_rules": [],
}


def record(**overrides):
    base = {
        "issue_id": "issue-1",
        "checker": "check_spelling",
        "confidence": "low",
        "field": "body_value",
        "original": "desmos",
        "proposed": "demos",
        "reasoning": "",
        "context_before": "",
        "context_after": "",
        "breadcrumb": "Unit 1",
        "node_label": "Lesson",
        "node_link": "https://example.test/node",
    }
    base.update(overrides)
    return base


def test_parser_extracts_and_deduplicates_issue():
    card = """
    <div class="issue-card" id="annotation-card-1"
         data-checker="check_spelling" data-confidence="low" data-field="body_value">
      <strong>Lesson</strong>: Sample Lesson&amp;
      <a href="https://example.test/node">Open</a>
      <span class="breadcrumb-pill">Unit 1</span>
      <div data-tab="text-diff">Use
        <span class="inline-del">desmos</span><span class="inline-ins">demos</span>
        here</div></div>
      <div class="aggregated-issue" data-checker="check_spelling">
        <span class="issue-original">desmos</span>
        <span class="issue-proposed">demos</span>
        <input data-comment-for="issue-1">
      </div>
    </div>
    """
    rows = parse_report_html(card + card)
    assert len(rows) == 1
    assert rows[0]["issue_id"] == "issue-1"
    assert rows[0]["original"] == "desmos"
    assert "Use" in rows[0]["context_before"]


def test_classifier_matches_core_skill_behavior():
    records = [
        record(),
        record(
            issue_id="math-1",
            checker="check_math",
            original="E(1)=20",
            proposed="E(1)=2.718282",
        ),
        record(
            issue_id="link-1",
            checker="check_links",
            original="https://example.test",
            proposed="[BROKEN]",
        ),
        record(
            issue_id="spacing-1",
            checker="check_spacing",
            original=".A",
            proposed=". A",
            context_before="HSA-APR",
        ),
        record(
            issue_id="unknown-spelling",
            original="traingle",
            proposed="triangle",
        ),
    ]
    rows = classify_records(records, BASE_RULES)
    statuses = {row["issue_id"]: row["status"] for row in rows}
    assert statuses == {
        "issue-1": "rejected",
        "math-1": "rejected",
        "link-1": "approved",
        "spacing-1": "rejected",
        "unknown-spelling": "needs_change",
    }


def test_exact_rule_overrides_default():
    rules = {
        **BASE_RULES,
        "exact_rules": [
            {
                "checker": "check_spelling",
                "field": "body_value",
                "original": "traingle",
                "proposed": "triangle",
                "status": "approved",
                "comment": "",
            }
        ],
    }
    rows = classify_records(
        [record(original="traingle", proposed="triangle")], rules
    )
    assert rows[0]["status"] == "approved"


def test_reviews_and_quoted_final_csv():
    rows = classify_records(
        [record(original="traingle", proposed="triangle")], BASE_RULES
    )
    reviews = {
        "issue-1": {"decision": "approved", "review_note": "Confirmed typo"}
    }
    assert review_progress(rows, reviews) == (1, 1)
    merged = apply_reviews(rows, reviews)
    assert merged[0]["status"] == "approved"
    parsed = list(csv.reader(io.StringIO(final_csv_bytes(merged).decode("utf-8"))))
    assert parsed[0] == ["issue_id", "status", "comment"]
    assert parsed[1] == ["issue-1", "approved", "Confirmed typo"]
    assert final_csv_bytes(merged).decode("utf-8").splitlines()[1].startswith('"')


def test_final_csv_removes_operational_review_comments():
    rows = classify_records(
        [record(original="traingle", proposed="triangle")], BASE_RULES
    )
    assert "confidence" in rows[0]["comment"].lower()
    parsed = list(
        csv.DictReader(io.StringIO(final_csv_bytes(rows).decode("utf-8")))
    )
    assert parsed[0]["comment"] == ""


def test_training_comment_keeps_only_reusable_qa_signals():
    assert (
        training_comment(
            "Human review confirmed that this is a valid curriculum-specific term.",
            checker="check_spelling",
            original="Desmos",
            proposed="demos",
        )
        == "this is a valid curriculum-specific term."
    )
    assert (
        training_comment(
            "No AI confidence is available. Needs human review.",
            checker="check_spelling",
            original="traingle",
            proposed="triangle",
        )
        == ""
    )
    assert (
        training_comment(
            'Use an ellipsis ("..."), not one period.',
            checker="check_punctuation",
            original="..",
            proposed=".",
        )
        == 'Use an ellipsis ("..."), not one period.'
    )
    assert (
        training_comment(
            "Math checker misread the expression; the proposed value is wrong.",
            checker="check_math",
        )
        == "Math checker misread the expression; the proposed value is wrong."
    )
    assert training_comment("Looks correct.", checker="check_spelling") == ""
    assert (
        training_comment(
            "Broken link needs replacement.",
            checker="check_links",
        )
        == ""
    )


def test_pattern_suggestion_and_promotion():
    rows = classify_records(
        [record(original="newjargon", proposed="new jargon")], BASE_RULES
    )
    reports = {
        "report-1": {
            "name": "Example",
            "rows": rows,
            "reviews": {
                "issue-1": {"decision": "rejected", "review_note": ""}
            },
        }
    }
    suggestions = build_pattern_suggestions(reports, BASE_RULES)
    assert len(suggestions) == 1
    assert suggestions[0]["suggestion_type"] == "protected spelling term"
    suggestions[0]["selected"] = True
    updated = promote_suggestions(BASE_RULES, suggestions)
    assert "newjargon" in updated["protected_spelling_terms"]


def test_shared_pattern_suggestion_includes_course_coverage():
    suggestions = build_shared_pattern_suggestions(
        [
            {
                "checker": "check_spelling",
                "field": "body_value",
                "original": "newjargon",
                "proposed": "new jargon",
                "status": "rejected",
                "comment": "",
                "human_decisions": 1,
                "human_reports": 1,
                "occurrences": 4,
                "course_coverage": 3,
                "reviewed_sources": "IM v.360 Grade 2",
                "seen_in_reports": (
                    "IM v.360 Grade 2, New Mexico Grade 2, Virginia Grade 2"
                ),
            }
        ],
        BASE_RULES,
    )
    assert suggestions[0]["suggestion_type"] == "protected spelling term"
    assert suggestions[0]["evidence_count"] == 1
    assert suggestions[0]["course_coverage"] == 3
    assert suggestions[0]["occurrences"] == 4


def test_bundled_rulebook_loads():
    rules = load_rules("rules/base_rules.json")
    assert rules["schema_version"] == 1
    assert "desmos" in rules["protected_spelling_terms"]
