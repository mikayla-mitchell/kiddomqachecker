from __future__ import annotations

import csv
import io
from pathlib import Path

from comment_policy import training_comment
from pattern_learning import (
    AUTOMATIC_RULE_SOURCE,
    automatic_suggestions,
    build_pattern_suggestions,
    build_shared_pattern_suggestions,
    promote_automatic_suggestions,
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


def test_parser_recovers_original_html_container():
    card = """
    <div class="issue-card" id="annotation-card-1"
         data-checker="check_capitalization" data-confidence="low"
         data-field="body_value">
      <strong>Lesson</strong>: Table&amp;
      <a href="https://example.test/node">Open</a>
      <div data-tab="original">&lt;table&gt;&lt;tr&gt;&lt;th&gt;temperature&lt;/th&gt;&lt;th&gt;sales&lt;/th&gt;&lt;/tr&gt;&lt;/table&gt;</div></div>
      <div data-tab="text-diff"><span class="inline-del">t</span><span class="inline-ins">T</span>emperature sales</div></div>
      <div class="aggregated-issue" data-checker="check_capitalization">
        <span class="issue-original">t</span>
        <span class="issue-proposed">T</span>
        <input data-comment-for="cap-1">
      </div>
    </div>
    """
    rows = parse_report_html(card)
    assert rows[0]["container_tag"] == "th"
    assert rows[0]["container_cell_has_prior_text"] is False


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


def test_new_structural_patterns_from_comparison_skill():
    records = [
        record(
            issue_id="latex-command",
            confidence="high",
            original="rightarrow",
            proposed="right arrow",
            context_before="Use \\",
        ),
        record(
            issue_id="placeholder",
            confidence="high",
            original="sss",
            proposed="ass",
        ),
        record(
            issue_id="bare-variable",
            checker="check_capitalization",
            original="x",
            proposed="X",
            context_after="=4",
        ),
        record(
            issue_id="defined-variable",
            checker="check_capitalization",
            original="n",
            proposed="N",
            context_after=" is the independent variable",
        ),
        record(
            issue_id="table-label",
            checker="check_capitalization",
            original="t",
            proposed="T",
            context_after="emperature",
            container_tag="th",
        ),
        record(
            issue_id="paragraph-start",
            checker="check_capitalization",
            original="t",
            proposed="T",
            context_after="he center",
            container_tag="p",
        ),
        record(
            issue_id="legacy-no-container-data",
            checker="check_capitalization",
            original="t",
            proposed="T",
            context_after="able",
        ),
        record(
            issue_id="parsed-no-markup-body",
            checker="check_capitalization",
            original="t",
            proposed="T",
            context_after="able",
            container_tag=None,
        ),
        record(
            issue_id="texas-standards",
            checker="check_proper_nouns",
            original="texas",
            proposed="Texas",
            context_after=" Essential Knowledge and Skills",
        ),
        record(
            issue_id="sentence-frame",
            checker="check_punctuation",
            original="..",
            proposed=".",
            context_after='" followed by a discussion note',
        ),
        record(
            issue_id="end-double-period",
            checker="check_punctuation",
            original="..",
            proposed=".",
            context_after="",
        ),
        record(
            issue_id="space-before-colon",
            checker="check_spacing",
            original=" :",
            proposed=":",
        ),
    ]
    rows = classify_records(records, BASE_RULES)
    statuses = {row["issue_id"]: row["status"] for row in rows}
    assert statuses == {
        "latex-command": "rejected",
        "placeholder": "rejected",
        "bare-variable": "rejected",
        "defined-variable": "rejected",
        "table-label": "rejected",
        "paragraph-start": "approved",
        "legacy-no-container-data": "needs_change",
        "parsed-no-markup-body": "rejected",
        "texas-standards": "approved",
        "sentence-frame": "rejected",
        "end-double-period": "needs_change",
        "space-before-colon": "approved",
    }


def test_math_earlier_in_context_does_not_hide_real_spacing_error():
    rows = classify_records(
        [
            record(
                checker="check_spacing",
                original=".W",
                proposed=". W",
                context_before=r"After \(x=4\), explain the result.",
                context_after="What happens next?",
            ),
            record(
                issue_id="dangling-latex",
                checker="check_spacing",
                original=";y",
                proposed="; y",
                context_before="Use \\",
            ),
        ],
        BASE_RULES,
    )
    assert [row["status"] for row in rows] == ["approved", "rejected"]


def test_base_rules_include_new_terms_and_verified_typos():
    rules_path = Path(__file__).parents[1] / "rules" / "base_rules.json"
    rules = load_rules(rules_path)
    assert {"preimage", "emoji", "teks", "subitize"} <= set(
        rules["protected_spelling_terms"]
    )
    typo_rows = classify_records(
        [
            record(original="unkown", proposed="unknown"),
            record(
                issue_id="number-line",
                original="numberline",
                proposed="number line",
            ),
        ],
        rules,
    )
    assert [row["status"] for row in typo_rows] == ["approved", "approved"]


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
    assert (
        training_comment(
            "Repeated-letter placeholder or math shorthand, not a misspelling.",
            checker="check_spelling",
            original="sss",
            proposed="ass",
        )
        == "Repeated-letter placeholder or math shorthand, not a misspelling."
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


def test_recurring_valid_term_is_learned_automatically():
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
                "occurrences": 2,
                "course_coverage": 2,
                "reviewed_sources": "IM v.360 Grade 2",
                "seen_in_reports": "IM v.360 Grade 2, New Mexico Grade 2",
            }
        ],
        BASE_RULES,
    )
    updated, promoted = promote_automatic_suggestions(BASE_RULES, suggestions)
    assert len(promoted) == 1
    assert "newjargon" in updated["protected_spelling_terms"]
    rows = classify_records(
        [record(original="newjargon", proposed="new jargon")], updated
    )
    assert rows[0]["status"] == "rejected"


def test_math_false_positive_is_learned_after_one_confirmed_decision():
    suggestions = build_shared_pattern_suggestions(
        [
            {
                "checker": "check_math",
                "field": "body_value",
                "original": "f(0)=6",
                "proposed": "f(0)=0",
                "status": "rejected",
                "comment": "Math expression is correct.",
                "human_decisions": 1,
                "human_reports": 1,
                "occurrences": 1,
                "course_coverage": 1,
                "reviewed_sources": "Algebra 1",
                "seen_in_reports": "Algebra 1",
            }
        ],
        BASE_RULES,
    )
    automatic = automatic_suggestions(suggestions)
    assert len(automatic) == 1
    updated, promoted = promote_automatic_suggestions(BASE_RULES, suggestions)
    assert len(promoted) == 1
    assert updated["exact_rules"][0]["source"] == AUTOMATIC_RULE_SOURCE


def test_spacing_rule_requires_consensus_across_reviewed_courses():
    evidence = {
        "checker": "check_spacing",
        "field": "body_value",
        "original": ":y",
        "proposed": ": y",
        "status": "rejected",
        "comment": "Math-expression boundary.",
        "human_decisions": 1,
        "human_reports": 1,
        "occurrences": 2,
        "course_coverage": 2,
        "reviewed_sources": "Algebra 1",
        "seen_in_reports": "Algebra 1, New Mexico Algebra 1",
    }
    suggestions = build_shared_pattern_suggestions([evidence], BASE_RULES)
    assert automatic_suggestions(suggestions) == []

    suggestions = build_shared_pattern_suggestions(
        [
            {
                **evidence,
                "human_decisions": 2,
                "human_reports": 2,
            }
        ],
        BASE_RULES,
    )
    assert len(automatic_suggestions(suggestions)) == 1


def test_needs_change_pattern_never_promotes_automatically():
    suggestions = build_shared_pattern_suggestions(
        [
            {
                "checker": "check_spacing",
                "field": "body_value",
                "original": "A",
                "proposed": " A",
                "status": "needs_change",
                "comment": "Use the correct spacing.",
                "human_decisions": 5,
                "human_reports": 5,
                "occurrences": 8,
                "course_coverage": 5,
                "reviewed_sources": "Course A, Course B",
                "seen_in_reports": "Course A, Course B",
            }
        ],
        BASE_RULES,
    )
    assert automatic_suggestions(suggestions) == []


def test_bundled_rulebook_loads():
    rules = load_rules("rules/base_rules.json")
    assert rules["schema_version"] == 1
    assert "desmos" in rules["protected_spelling_terms"]
