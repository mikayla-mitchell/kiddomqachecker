from __future__ import annotations

import sqlite3

from decision_memory import (
    export_memory_bytes,
    import_memory_bytes,
    initialize_memory,
    issue_signature,
    library_stats,
    list_report_library,
    load_draft_reviews,
    load_report_jira_link,
    load_report_snapshot,
    match_report_rows,
    memory_stats,
    publish_report_reviews,
    report_similarity,
    save_draft_reviews,
    save_report_jira_link,
    shared_pattern_evidence,
    store_report_snapshot,
)


def review_row(
    issue_id: str,
    *,
    before: str = "",
    after: str = "",
    breadcrumb: str = "Unit 2 > Lesson 4",
):
    return {
        "issue_id": issue_id,
        "status": "needs_change",
        "checker": "check_capitalization",
        "field": "body_value",
        "original": "i",
        "proposed": "I",
        "context_before": before,
        "context_after": after,
        "breadcrumb": breadcrumb,
        "node_label": "Student Task",
    }


def test_exact_match_ignores_issue_id_urls_and_uuids(tmp_path):
    database = tmp_path / "memory.sqlite3"
    source = review_row(
        "source-1",
        before=(
            "Open https://example.test/old and item "
            "44365624-cc6f-11ef-a9a4-066a39b724af before "
        ),
        after=" the same curriculum sentence continues here.",
    )
    publish_report_reviews(
        database,
        "v360-grade-2",
        "IM v.360 Grade 2",
        [source],
        {"source-1": {"decision": "rejected", "review_note": ""}},
    )

    target = review_row(
        "new-mexico-99",
        before=(
            "Open https://different.test/new and item "
            "ee4874c3-24fc-11ef-aff7-02fe2bddb0a9 before "
        ),
        after=" the same curriculum sentence continues here.",
    )
    matches = match_report_rows(database, [target])
    assert matches["new-mexico-99"]["decision"] == "rejected"
    assert matches["new-mexico-99"]["memory_match"] == "exact"
    assert matches["new-mexico-99"]["memory_source"] == "IM v.360 Grade 2"


def test_near_match_requires_same_core_and_high_context_similarity(tmp_path):
    database = tmp_path / "memory.sqlite3"
    long_before = (
        "Students compare two strategies and explain why the relationship "
        "between the quantities remains constant throughout the activity. "
    )
    source = review_row(
        "source-1",
        before=long_before,
        after="They record the conclusion in their workbook.",
    )
    publish_report_reviews(
        database,
        "v360-grade-2",
        "IM v.360 Grade 2",
        [source],
        {"source-1": {"decision": "approved", "review_note": ""}},
    )
    target = review_row(
        "nm-1",
        before=long_before.replace("activity", "New Mexico activity"),
        after="They record the conclusion in their workbook.",
    )
    matches = match_report_rows(database, [target])
    assert matches["nm-1"]["decision"] == "approved"
    assert matches["nm-1"]["memory_match"] == "near"
    assert matches["nm-1"]["memory_score"] >= 0.97


def test_published_memory_sanitizes_operational_notes(tmp_path):
    database = tmp_path / "memory.sqlite3"
    row = review_row(
        "source-1",
        before="A long context identifies the same finding across reports.",
        after="The following context confirms the same curriculum location.",
    )
    publish_report_reviews(
        database,
        "course-a",
        "Course A",
        [row],
        {
            "source-1": {
                "decision": "rejected",
                "review_note": (
                    "Human review confirmed that this is a valid curriculum-specific term."
                ),
            }
        },
    )
    matched = match_report_rows(
        database, [{**row, "issue_id": "target-1"}]
    )
    assert (
        matched["target-1"]["review_note"]
        == "this is a valid curriculum-specific term."
    )


def test_legacy_memory_note_is_filtered_when_reused(tmp_path):
    database = tmp_path / "memory.sqlite3"
    row = review_row(
        "source-1",
        before="A sufficiently long context anchors the legacy decision.",
        after="The same after-context appears in the later report.",
    )
    publish_report_reviews(
        database,
        "course-a",
        "Course A",
        [row],
        {"source-1": {"decision": "rejected", "review_note": ""}},
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE observations SET note = ?",
            ("No AI confidence; needs human review.",),
        )
    matched = match_report_rows(
        database, [{**row, "issue_id": "target-1"}]
    )
    assert matched["target-1"]["decision"] == "rejected"
    assert matched["target-1"]["review_note"] == ""


def test_conflicting_prior_reviews_are_not_reused(tmp_path):
    database = tmp_path / "memory.sqlite3"
    row = review_row(
        "shared-issue",
        before="A sufficiently long identical sentence appears before the finding.",
        after="A sufficiently long identical sentence appears after the finding.",
    )
    publish_report_reviews(
        database,
        "course-a",
        "Course A",
        [row],
        {"shared-issue": {"decision": "approved", "review_note": ""}},
    )
    publish_report_reviews(
        database,
        "course-b",
        "Course B",
        [row],
        {"shared-issue": {"decision": "rejected", "review_note": ""}},
    )
    target = {**row, "issue_id": "target"}
    assert match_report_rows(database, [target]) == {}
    assert memory_stats(database)["conflicts"] == 1


def test_memory_export_import_round_trip(tmp_path):
    source_database = tmp_path / "source.sqlite3"
    target_database = tmp_path / "target.sqlite3"
    row = review_row(
        "issue-1",
        before="A long context before the issue makes this finding reusable.",
        after="A long context after the issue confirms the same content.",
    )
    publish_report_reviews(
        source_database,
        "course-a",
        "Course A",
        [row],
        {"issue-1": {"decision": "needs_change", "review_note": "Use ellipsis"}},
    )
    result = import_memory_bytes(
        target_database, export_memory_bytes(source_database)
    )
    assert result["imported"] == 1
    assert memory_stats(target_database) == {
        "observations": 1,
        "findings": 1,
        "reports": 1,
        "conflicts": 0,
    }
    matched = match_report_rows(
        target_database, [{**row, "issue_id": "different-id"}]
    )
    assert matched["different-id"]["decision"] == "needs_change"
    assert matched["different-id"]["review_note"] == "Use ellipsis"


def test_unanchored_findings_are_not_fingerprinted():
    row = review_row("issue-1", breadcrumb="")
    row["node_label"] = ""
    assert issue_signature(row) is None


def test_course_prefix_does_not_change_location_fingerprint():
    source = review_row(
        "source",
        breadcrumb="Unit 2 > Check Your Readiness",
    )
    target = review_row(
        "target",
        breadcrumb="New Mexico Grade 2 > Unit 2 > Check Your Readiness",
    )
    assert issue_signature(source)["fingerprint"] == issue_signature(target)["fingerprint"]


def test_report_library_persists_snapshots_drafts_and_overlap(tmp_path):
    database = tmp_path / "memory.sqlite3"
    source = review_row(
        "source-1",
        before="Students compare the same quantities in this long shared context.",
        after="They explain the relationship using a complete sentence.",
    )
    target = {
        **source,
        "issue_id": "target-1",
        "breadcrumb": "New Mexico Grade 2 > Unit 2 > Lesson 4",
    }
    store_report_snapshot(
        database,
        "report-a",
        "IM v.360 Grade 2",
        "v360.html",
        [source],
        [source],
        "test",
    )
    store_report_snapshot(
        database,
        "report-b",
        "New Mexico Grade 2",
        "new-mexico.html",
        [target],
        [target],
        "test",
    )
    save_draft_reviews(
        database,
        "report-a",
        {"source-1": {"decision": "rejected", "review_note": ""}},
    )

    assert library_stats(database) == {
        "reports": 2,
        "findings": 2,
        "recurring_findings": 1,
        "draft_reviews": 1,
    }
    loaded = load_report_snapshot(database, "report-a")
    assert loaded["filename"] == "v360.html"
    assert loaded["records"][0]["issue_id"] == "source-1"
    assert load_draft_reviews(database, "report-a")["source-1"]["decision"] == "rejected"
    listed = {item["report_id"]: item for item in list_report_library(database)}
    assert listed["report-a"]["draft_reviews"] == 1
    overlap = report_similarity(database, "report-b")
    assert overlap[0]["report_id"] == "report-a"
    assert overlap[0]["exact_findings"] == 1
    assert overlap[0]["exact_overlap"] == 1.0


def test_shared_pattern_evidence_uses_unreviewed_reports_only_for_coverage(tmp_path):
    database = tmp_path / "memory.sqlite3"
    source = review_row(
        "source-1",
        before="A long repeated context appears before this finding in both courses.",
        after="A long repeated context appears after this finding in both courses.",
    )
    target = {**source, "issue_id": "target-1"}
    store_report_snapshot(
        database,
        "report-a",
        "IM v.360 Grade 2",
        "v360.html",
        [source],
        [source],
        "test",
    )
    store_report_snapshot(
        database,
        "report-b",
        "New Mexico Grade 2",
        "new-mexico.html",
        [target],
        [target],
        "test",
    )
    publish_report_reviews(
        database,
        "report-a",
        "IM v.360 Grade 2",
        [source],
        {"source-1": {"decision": "rejected", "review_note": ""}},
    )
    evidence = shared_pattern_evidence(database)
    assert len(evidence) == 1
    assert evidence[0]["human_decisions"] == 1
    assert evidence[0]["human_reports"] == 1
    assert evidence[0]["occurrences"] == 2
    assert evidence[0]["course_coverage"] == 2


def test_full_shared_memory_export_import_round_trip(tmp_path):
    source_database = tmp_path / "source.sqlite3"
    target_database = tmp_path / "target.sqlite3"
    row = review_row(
        "issue-1",
        before="A long context makes this full report snapshot portable.",
        after="The matching after-context is also retained in structured form.",
    )
    store_report_snapshot(
        source_database,
        "report-a",
        "Course A",
        "course-a.html",
        [row],
        [row],
        "test",
    )
    save_draft_reviews(
        source_database,
        "report-a",
        {"issue-1": {"decision": "approved", "review_note": ""}},
    )
    save_report_jira_link(
        source_database,
        "report-a",
        {
            "key": "CURR-42",
            "summary": "Review Course A",
            "browse_url": "https://example.atlassian.net/browse/CURR-42",
        },
        "9001",
    )
    result = import_memory_bytes(
        target_database, export_memory_bytes(source_database)
    )
    assert result == {
        "imported": 0,
        "reports": 1,
        "findings": 1,
        "drafts": 1,
    }
    assert library_stats(target_database)["reports"] == 1
    assert load_report_snapshot(target_database, "report-a")["records"][0][
        "issue_id"
    ] == "issue-1"
    jira_link = load_report_jira_link(target_database, "report-a")
    assert jira_link["key"] == "CURR-42"
    assert jira_link["summary"] == "Review Course A"
    assert jira_link["browse_url"].endswith("/browse/CURR-42")
    assert jira_link["attachment_id"] == "9001"


def test_schema_one_database_migrates_in_place(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES('schema_version', '1')"
        )
    initialize_memory(database)
    assert library_stats(database)["reports"] == 0
    with sqlite3.connect(database) as connection:
        version = connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()[0]
    assert version == "2"
