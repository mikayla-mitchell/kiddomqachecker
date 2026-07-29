from __future__ import annotations

import hashlib
import html
import io
import json
import re
import sqlite3
import uuid
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping

from comment_policy import training_comment
from qa_engine import parse_decision


SCHEMA_VERSION = 3
FINGERPRINT_VERSION = 1
DEFAULT_NEAR_MATCH_THRESHOLD = 0.97
VALID_STATUSES = {"approved", "rejected", "needs_change"}
OBSERVATION_FIELDS = (
    "observation_id",
    "fingerprint",
    "core_key",
    "anchor_kind",
    "anchor",
    "checker",
    "field_name",
    "original",
    "proposed",
    "status",
    "note",
    "source_report",
    "source_report_id",
    "issue_id",
    "updated_at",
)
REPORT_FIELDS = (
    "report_id",
    "report_name",
    "filename",
    "rules_version",
    "finding_count",
    "reviewable_count",
    "first_uploaded_at",
    "last_uploaded_at",
    "upload_count",
)
REPORT_FINDING_FIELDS = (
    "finding_id",
    "report_id",
    "finding_order",
    "issue_id",
    "fingerprint",
    "core_key",
    "anchor_kind",
    "anchor",
    "checker",
    "field_name",
    "original",
    "proposed",
    "initial_status",
    "initial_comment",
    "record_json",
)
DRAFT_REVIEW_FIELDS = (
    "report_id",
    "issue_id",
    "decision",
    "note",
    "updated_at",
)
JIRA_LINK_FIELDS = (
    "report_id",
    "issue_key",
    "issue_summary",
    "issue_url",
    "attachment_id",
    "updated_at",
)
REVIEW_EVENT_FIELDS = (
    "event_id",
    "report_id",
    "issue_id",
    "action",
    "decision",
    "reviewer_email",
    "reviewer_name",
    "detail_json",
    "occurred_at",
)

URL_RE = re.compile(r"https?://\S+", re.I)
UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.I,
)
WHITESPACE_RE = re.compile(r"\s+")


class DecisionMemoryError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_match_text(value: Any) -> str:
    text = html.unescape(str(value or "")).casefold()
    text = URL_RE.sub(" <url> ", text)
    text = UUID_RE.sub(" <uuid> ", text)
    text = text.replace("\u00a0", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_breadcrumb(value: Any) -> str:
    segments = [
        normalize_match_text(segment)
        for segment in str(value or "").split(">")
        if normalize_match_text(segment)
    ]
    for index, segment in enumerate(segments):
        if re.search(r"\b(?:unit|section|lesson|check|assessment|module)\b", segment):
            segments = segments[index:]
            break
    # Course/edition names are commonly prepended as extra hierarchy levels.
    # The final Unit/Section/Lesson path is the portable location signal.
    return " > ".join(segments[-4:])


def _anchor_for_row(row: Mapping[str, Any]) -> tuple[str, str] | None:
    before = normalize_match_text(row.get("context_before"))
    after = normalize_match_text(row.get("context_after"))
    context_anchor = f"{before} <change> {after}".strip()
    visible_context = f"{before}{after}".strip()
    if len(visible_context) >= 20:
        return "context", context_anchor

    breadcrumb = normalize_breadcrumb(row.get("breadcrumb"))
    node_label = normalize_match_text(row.get("node_label"))
    location_anchor = f"{breadcrumb} | {node_label}".strip(" |")
    if len(location_anchor) >= 8:
        return "location", location_anchor
    return None


def issue_core_signature(row: Mapping[str, Any]) -> dict[str, str]:
    checker = normalize_match_text(row.get("checker"))
    field = normalize_match_text(row.get("field"))
    original = normalize_match_text(row.get("original"))
    proposed = normalize_match_text(row.get("proposed"))
    core_payload = json.dumps(
        [FINGERPRINT_VERSION, checker, field, original, proposed],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    core_key = hashlib.sha256(core_payload.encode("utf-8")).hexdigest()
    return {
        "core_key": core_key,
        "checker": checker,
        "field": field,
        "original": original,
        "proposed": proposed,
    }


def issue_signature(row: Mapping[str, Any]) -> dict[str, str] | None:
    anchor_data = _anchor_for_row(row)
    if not anchor_data:
        return None
    anchor_kind, anchor = anchor_data
    core = issue_core_signature(row)
    fingerprint_payload = json.dumps(
        [FINGERPRINT_VERSION, core["core_key"], anchor_kind, anchor],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
    return {
        "fingerprint": fingerprint,
        "anchor_kind": anchor_kind,
        "anchor": anchor,
        **core,
    }


def _connect(path: str | Path) -> sqlite3.Connection:
    database = Path(path).expanduser()
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS observations (
            observation_id TEXT PRIMARY KEY,
            fingerprint TEXT NOT NULL,
            core_key TEXT NOT NULL,
            anchor_kind TEXT NOT NULL,
            anchor TEXT NOT NULL,
            checker TEXT NOT NULL,
            field_name TEXT NOT NULL,
            original TEXT NOT NULL,
            proposed TEXT NOT NULL,
            status TEXT NOT NULL,
            note TEXT NOT NULL,
            source_report TEXT NOT NULL,
            source_report_id TEXT NOT NULL,
            issue_id TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_observations_fingerprint
            ON observations(fingerprint);
        CREATE INDEX IF NOT EXISTS idx_observations_core_key
            ON observations(core_key);
        CREATE INDEX IF NOT EXISTS idx_observations_source
            ON observations(source_report_id);

        CREATE TABLE IF NOT EXISTS stored_reports (
            report_id TEXT PRIMARY KEY,
            report_name TEXT NOT NULL,
            filename TEXT NOT NULL,
            rules_version TEXT NOT NULL,
            finding_count INTEGER NOT NULL,
            reviewable_count INTEGER NOT NULL,
            first_uploaded_at TEXT NOT NULL,
            last_uploaded_at TEXT NOT NULL,
            upload_count INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS stored_findings (
            finding_id TEXT PRIMARY KEY,
            report_id TEXT NOT NULL,
            finding_order INTEGER NOT NULL,
            issue_id TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            core_key TEXT NOT NULL,
            anchor_kind TEXT NOT NULL,
            anchor TEXT NOT NULL,
            checker TEXT NOT NULL,
            field_name TEXT NOT NULL,
            original TEXT NOT NULL,
            proposed TEXT NOT NULL,
            initial_status TEXT NOT NULL,
            initial_comment TEXT NOT NULL,
            record_json TEXT NOT NULL,
            FOREIGN KEY(report_id) REFERENCES stored_reports(report_id)
                ON DELETE CASCADE,
            UNIQUE(report_id, issue_id)
        );

        CREATE INDEX IF NOT EXISTS idx_stored_findings_report
            ON stored_findings(report_id);
        CREATE INDEX IF NOT EXISTS idx_stored_findings_fingerprint
            ON stored_findings(fingerprint);
        CREATE INDEX IF NOT EXISTS idx_stored_findings_core_key
            ON stored_findings(core_key);

        CREATE TABLE IF NOT EXISTS draft_reviews (
            report_id TEXT NOT NULL,
            issue_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            note TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(report_id, issue_id),
            FOREIGN KEY(report_id) REFERENCES stored_reports(report_id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_draft_reviews_report
            ON draft_reviews(report_id);

        CREATE TABLE IF NOT EXISTS report_jira_links (
            report_id TEXT PRIMARY KEY,
            issue_key TEXT NOT NULL,
            issue_summary TEXT NOT NULL,
            issue_url TEXT NOT NULL,
            attachment_id TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(report_id) REFERENCES stored_reports(report_id)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS review_events (
            event_id TEXT PRIMARY KEY,
            report_id TEXT NOT NULL,
            issue_id TEXT NOT NULL,
            action TEXT NOT NULL,
            decision TEXT NOT NULL,
            reviewer_email TEXT NOT NULL,
            reviewer_name TEXT NOT NULL,
            detail_json TEXT NOT NULL,
            occurred_at TEXT NOT NULL,
            FOREIGN KEY(report_id) REFERENCES stored_reports(report_id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_review_events_report
            ON review_events(report_id, occurred_at);
        CREATE INDEX IF NOT EXISTS idx_review_events_reviewer
            ON review_events(reviewer_email, occurred_at);
        """
    )
    stored_schema = connection.execute(
        "SELECT value FROM metadata WHERE key='schema_version'"
    ).fetchone()
    stored_version = int(stored_schema["value"]) if stored_schema else SCHEMA_VERSION
    if stored_version not in {1, 2, SCHEMA_VERSION}:
        connection.close()
        raise DecisionMemoryError("Unsupported decision-memory database schema.")
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('fingerprint_version', ?)",
        (str(FINGERPRINT_VERSION),),
    )
    connection.commit()
    return connection


def initialize_memory(path: str | Path) -> None:
    with closing(_connect(path)):
        pass


def _finding_id(report_id: str, issue_id: str) -> str:
    payload = f"{report_id}\x1f{issue_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def store_report_snapshot(
    path: str | Path,
    report_id: str,
    report_name: str,
    filename: str,
    records: Iterable[Mapping[str, Any]],
    rows: Iterable[Mapping[str, Any]],
    rules_version: str,
) -> dict[str, int]:
    records_list = [dict(record) for record in records]
    rows_list = [dict(row) for row in rows]
    records_by_id = {
        str(record.get("issue_id") or ""): record for record in records_list
    }
    now = _utc_now()
    matchable = 0
    with closing(_connect(path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO stored_reports(
                report_id, report_name, filename, rules_version,
                finding_count, reviewable_count, first_uploaded_at,
                last_uploaded_at, upload_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(report_id) DO UPDATE SET
                report_name=excluded.report_name,
                filename=excluded.filename,
                rules_version=excluded.rules_version,
                finding_count=excluded.finding_count,
                reviewable_count=excluded.reviewable_count,
                last_uploaded_at=excluded.last_uploaded_at,
                upload_count=stored_reports.upload_count + 1
            """,
            (
                report_id,
                report_name,
                filename,
                rules_version,
                len(rows_list),
                sum(row.get("status") == "needs_change" for row in rows_list),
                now,
                now,
            ),
        )
        connection.execute(
            "DELETE FROM stored_findings WHERE report_id = ?", (report_id,)
        )
        finding_values = []
        for finding_order, row in enumerate(rows_list):
            issue_id = str(row.get("issue_id") or "")
            signature = issue_signature(row)
            core = signature or issue_core_signature(row)
            if signature:
                matchable += 1
            record = records_by_id.get(issue_id, row)
            finding_values.append(
                (
                    _finding_id(report_id, issue_id),
                    report_id,
                    finding_order,
                    issue_id,
                    signature["fingerprint"] if signature else "",
                    core["core_key"],
                    signature["anchor_kind"] if signature else "",
                    signature["anchor"] if signature else "",
                    core["checker"],
                    core["field"],
                    str(row.get("original") or ""),
                    str(row.get("proposed") or ""),
                    str(row.get("status") or ""),
                    str(row.get("comment") or ""),
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    ),
                )
            )
        connection.executemany(
            """
            INSERT INTO stored_findings(
                finding_id, report_id, finding_order, issue_id, fingerprint,
                core_key, anchor_kind, anchor, checker, field_name, original,
                proposed, initial_status, initial_comment, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            finding_values,
        )
    return {"stored": len(rows_list), "matchable": matchable}


def load_report_snapshot(path: str | Path, report_id: str) -> dict[str, Any]:
    with closing(_connect(path)) as connection:
        report = connection.execute(
            "SELECT * FROM stored_reports WHERE report_id = ?", (report_id,)
        ).fetchone()
        if not report:
            raise DecisionMemoryError("The saved report could not be found.")
        finding_rows = connection.execute(
            """
            SELECT record_json
            FROM stored_findings
            WHERE report_id = ?
            ORDER BY finding_order
            """,
            (report_id,),
        ).fetchall()
    try:
        records = [json.loads(str(row["record_json"])) for row in finding_rows]
    except json.JSONDecodeError as error:
        raise DecisionMemoryError("A saved report snapshot is corrupted.") from error
    return {
        "report_id": str(report["report_id"]),
        "name": str(report["report_name"]),
        "filename": str(report["filename"]),
        "records": records,
        "rules_version": str(report["rules_version"]),
    }


def save_draft_reviews(
    path: str | Path,
    report_id: str,
    reviews: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    saved = 0
    cleared = 0
    now = _utc_now()
    with closing(_connect(path)) as connection, connection:
        for issue_id, review in reviews.items():
            decision, note = parse_decision(
                str(review.get("decision") or ""),
                str(review.get("review_note") or ""),
            )
            if not decision:
                connection.execute(
                    "DELETE FROM draft_reviews WHERE report_id = ? AND issue_id = ?",
                    (report_id, str(issue_id)),
                )
                cleared += 1
                continue
            connection.execute(
                """
                INSERT INTO draft_reviews(
                    report_id, issue_id, decision, note, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(report_id, issue_id) DO UPDATE SET
                    decision=excluded.decision,
                    note=excluded.note,
                    updated_at=excluded.updated_at
                """,
                (report_id, str(issue_id), decision, note or "", now),
            )
            saved += 1
    return {"saved": saved, "cleared": cleared}


def load_draft_reviews(
    path: str | Path, report_id: str
) -> dict[str, dict[str, Any]]:
    with closing(_connect(path)) as connection:
        rows = connection.execute(
            """
            SELECT issue_id, decision, note, updated_at
            FROM draft_reviews
            WHERE report_id = ?
            """,
            (report_id,),
        ).fetchall()
    return {
        str(row["issue_id"]): {
            "decision": str(row["decision"]),
            "review_note": str(row["note"]),
            "draft_saved": True,
            "draft_updated_at": str(row["updated_at"]),
        }
        for row in rows
    }


def save_report_jira_link(
    path: str | Path,
    report_id: str,
    issue: Mapping[str, Any],
    attachment_id: str = "",
) -> None:
    issue_key = str(issue.get("key") or "").strip()
    issue_url = str(issue.get("browse_url") or "").strip()
    if not issue_key or not issue_url:
        raise DecisionMemoryError(
            "A Jira report link requires an issue key and browse URL."
        )
    with closing(_connect(path)) as connection, connection:
        try:
            connection.execute(
                """
                INSERT INTO report_jira_links(
                    report_id, issue_key, issue_summary, issue_url,
                    attachment_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(report_id) DO UPDATE SET
                    issue_key=excluded.issue_key,
                    issue_summary=excluded.issue_summary,
                    issue_url=excluded.issue_url,
                    attachment_id=excluded.attachment_id,
                    updated_at=excluded.updated_at
                """,
                (
                    report_id,
                    issue_key,
                    str(issue.get("summary") or ""),
                    issue_url,
                    str(attachment_id or ""),
                    _utc_now(),
                ),
            )
        except sqlite3.IntegrityError as error:
            raise DecisionMemoryError(
                "Save the report snapshot before linking it to Jira."
            ) from error


def load_report_jira_link(
    path: str | Path, report_id: str
) -> dict[str, str] | None:
    with closing(_connect(path)) as connection:
        row = connection.execute(
            "SELECT * FROM report_jira_links WHERE report_id = ?",
            (report_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "key": str(row["issue_key"]),
        "summary": str(row["issue_summary"]),
        "browse_url": str(row["issue_url"]),
        "attachment_id": str(row["attachment_id"]),
        "updated_at": str(row["updated_at"]),
    }


def record_review_events(
    path: str | Path,
    report_id: str,
    action: str,
    reviewer: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]],
) -> int:
    reviewer_email = str(reviewer.get("email") or "").strip().casefold()
    reviewer_name = str(reviewer.get("name") or reviewer_email or "Unknown reviewer")
    normalized_action = str(action or "").strip()
    if not normalized_action:
        raise DecisionMemoryError("A review event requires an action.")
    if not reviewer_email:
        raise DecisionMemoryError("A review event requires a reviewer email.")

    values = []
    now = _utc_now()
    for event in events:
        detail = event.get("detail")
        detail = detail if isinstance(detail, Mapping) else {}
        values.append(
            (
                uuid.uuid4().hex,
                report_id,
                str(event.get("issue_id") or ""),
                normalized_action,
                str(event.get("decision") or ""),
                reviewer_email,
                reviewer_name,
                json.dumps(detail, ensure_ascii=False, separators=(",", ":"), default=str),
                now,
            )
        )
    if not values:
        return 0
    with closing(_connect(path)) as connection, connection:
        try:
            connection.executemany(
                """
                INSERT INTO review_events(
                    event_id, report_id, issue_id, action, decision,
                    reviewer_email, reviewer_name, detail_json, occurred_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        except sqlite3.IntegrityError as error:
            raise DecisionMemoryError(
                "Save the report snapshot before recording reviewer activity."
            ) from error
    return len(values)


def list_report_review_activity(
    path: str | Path, report_id: str, limit: int = 50
) -> list[dict[str, Any]]:
    with closing(_connect(path)) as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM review_events
            WHERE report_id = ?
            ORDER BY occurred_at DESC, event_id DESC
            LIMIT ?
            """,
            (report_id, max(1, int(limit))),
        ).fetchall()
    activity = []
    for row in rows:
        item = dict(row)
        try:
            item["detail"] = json.loads(str(item.pop("detail_json")))
        except json.JSONDecodeError:
            item["detail"] = {}
        activity.append(item)
    return activity


def list_report_library(path: str | Path) -> list[dict[str, Any]]:
    with closing(_connect(path)) as connection:
        rows = connection.execute(
            """
            SELECT
                report_id,
                report_name,
                filename,
                rules_version,
                finding_count,
                reviewable_count,
                first_uploaded_at,
                last_uploaded_at,
                upload_count,
                (
                    SELECT COUNT(*)
                    FROM draft_reviews d
                    WHERE d.report_id = stored_reports.report_id
                ) AS draft_reviews,
                (
                    SELECT COUNT(*)
                    FROM observations o
                    WHERE o.source_report_id = stored_reports.report_id
                ) AS published_reviews
            FROM stored_reports
            ORDER BY last_uploaded_at DESC, report_name
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _observation_id(source_report_id: str, issue_id: str) -> str:
    payload = f"{source_report_id}\x1f{issue_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def publish_report_reviews(
    path: str | Path,
    source_report_id: str,
    source_report: str,
    rows: Iterable[Mapping[str, Any]],
    reviews: Mapping[str, Mapping[str, str]],
) -> dict[str, int]:
    published = 0
    skipped_blank = 0
    skipped_unmatchable = 0
    now = _utc_now()
    with closing(_connect(path)) as connection, connection:
        for row in rows:
            if row.get("status") != "needs_change":
                continue
            issue_id = str(row.get("issue_id") or "")
            review = reviews.get(issue_id, {})
            status, note = parse_decision(
                str(review.get("decision") or ""),
                str(review.get("review_note") or ""),
            )
            if not status:
                skipped_blank += 1
                continue
            note = training_comment(
                note,
                status=status,
                checker=row.get("checker"),
                original=row.get("original"),
                proposed=row.get("proposed"),
            )
            signature = issue_signature(row)
            if not signature:
                skipped_unmatchable += 1
                continue
            connection.execute(
                """
                INSERT INTO observations(
                    observation_id, fingerprint, core_key, anchor_kind, anchor,
                    checker, field_name, original, proposed, status, note,
                    source_report, source_report_id, issue_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(observation_id) DO UPDATE SET
                    fingerprint=excluded.fingerprint,
                    core_key=excluded.core_key,
                    anchor_kind=excluded.anchor_kind,
                    anchor=excluded.anchor,
                    checker=excluded.checker,
                    field_name=excluded.field_name,
                    original=excluded.original,
                    proposed=excluded.proposed,
                    status=excluded.status,
                    note=excluded.note,
                    source_report=excluded.source_report,
                    updated_at=excluded.updated_at
                """,
                (
                    _observation_id(source_report_id, issue_id),
                    signature["fingerprint"],
                    signature["core_key"],
                    signature["anchor_kind"],
                    signature["anchor"],
                    signature["checker"],
                    signature["field"],
                    signature["original"],
                    signature["proposed"],
                    status,
                    note or "",
                    source_report,
                    source_report_id,
                    issue_id,
                    now,
                ),
            )
            published += 1
    return {
        "published": published,
        "skipped_blank": skipped_blank,
        "skipped_unmatchable": skipped_unmatchable,
    }


def _consensus(observations: Iterable[sqlite3.Row]) -> dict[str, Any] | None:
    rows = list(observations)
    if not rows:
        return None
    statuses = {str(row["status"]) for row in rows}
    if len(statuses) != 1:
        return None
    status = statuses.pop()
    if status not in VALID_STATUSES:
        return None
    notes = {
        filtered
        for row in rows
        if (
            filtered := training_comment(
                row["note"],
                status=status,
                checker=row["checker"],
                original=row["original"],
                proposed=row["proposed"],
            )
        )
    }
    if status == "needs_change" and len(notes) != 1:
        return None
    note = next(iter(notes)) if len(notes) == 1 else ""
    return {
        "decision": status,
        "review_note": note,
        "memory_source": ", ".join(
            sorted({str(row["source_report"]) for row in rows})
        ),
        "memory_evidence": len(rows),
    }


def _fetch_relevant_observations(
    connection: sqlite3.Connection, core_keys: Iterable[str]
) -> list[sqlite3.Row]:
    unique = sorted(set(core_keys))
    results: list[sqlite3.Row] = []
    for start in range(0, len(unique), 800):
        batch = unique[start : start + 800]
        placeholders = ",".join("?" for _ in batch)
        results.extend(
            connection.execute(
                f"SELECT * FROM observations WHERE core_key IN ({placeholders})",
                batch,
            ).fetchall()
        )
    return results


def match_report_rows(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    near_match_threshold: float = DEFAULT_NEAR_MATCH_THRESHOLD,
) -> dict[str, dict[str, Any]]:
    candidate_rows = [
        dict(row) for row in rows if row.get("status") == "needs_change"
    ]
    signatures = {
        str(row["issue_id"]): issue_signature(row) for row in candidate_rows
    }
    signatures = {
        issue_id: signature
        for issue_id, signature in signatures.items()
        if signature is not None
    }
    if not signatures:
        return {}

    with closing(_connect(path)) as connection:
        observations = _fetch_relevant_observations(
            connection, [signature["core_key"] for signature in signatures.values()]
        )

    by_fingerprint: dict[str, list[sqlite3.Row]] = defaultdict(list)
    by_core: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for observation in observations:
        by_fingerprint[str(observation["fingerprint"])].append(observation)
        by_core[str(observation["core_key"])].append(observation)

    matches: dict[str, dict[str, Any]] = {}
    for issue_id, signature in signatures.items():
        exact = _consensus(by_fingerprint.get(signature["fingerprint"], []))
        if exact:
            matches[issue_id] = {
                **exact,
                "memory_match": "exact",
                "memory_score": 1.0,
            }
            continue

        scored_groups: dict[str, dict[str, Any]] = {}
        for observation in by_core.get(signature["core_key"], []):
            if observation["anchor_kind"] != signature["anchor_kind"]:
                continue
            score = SequenceMatcher(
                None, signature["anchor"], str(observation["anchor"])
            ).ratio()
            if score < near_match_threshold:
                continue
            fingerprint = str(observation["fingerprint"])
            group = scored_groups.setdefault(
                fingerprint, {"score": score, "rows": []}
            )
            group["score"] = max(group["score"], score)
            group["rows"].append(observation)
        if not scored_groups:
            continue

        best_score = max(group["score"] for group in scored_groups.values())
        best_groups = [
            group
            for group in scored_groups.values()
            if group["score"] >= best_score - 0.005
        ]
        near_consensus = _consensus(
            row for group in best_groups for row in group["rows"]
        )
        if near_consensus:
            matches[issue_id] = {
                **near_consensus,
                "memory_match": "near",
                "memory_score": round(best_score, 4),
            }
    return matches


def report_similarity(
    path: str | Path, report_id: str, limit: int = 12
) -> list[dict[str, Any]]:
    with closing(_connect(path)) as connection:
        current_counts = connection.execute(
            """
            SELECT
                COUNT(DISTINCT NULLIF(fingerprint, '')) AS fingerprints,
                COUNT(DISTINCT core_key) AS core_keys
            FROM stored_findings
            WHERE report_id = ?
            """,
            (report_id,),
        ).fetchone()
        exact_rows = connection.execute(
            """
            WITH current_fingerprints AS (
                SELECT DISTINCT fingerprint
                FROM stored_findings
                WHERE report_id = ? AND fingerprint != ''
            )
            SELECT other.report_id, COUNT(DISTINCT other.fingerprint) AS matches
            FROM stored_findings other
            JOIN current_fingerprints current
                ON current.fingerprint = other.fingerprint
            WHERE other.report_id != ?
            GROUP BY other.report_id
            """,
            (report_id, report_id),
        ).fetchall()
        core_rows = connection.execute(
            """
            WITH current_cores AS (
                SELECT DISTINCT core_key
                FROM stored_findings
                WHERE report_id = ?
            )
            SELECT other.report_id, COUNT(DISTINCT other.core_key) AS matches
            FROM stored_findings other
            JOIN current_cores current
                ON current.core_key = other.core_key
            WHERE other.report_id != ?
            GROUP BY other.report_id
            """,
            (report_id, report_id),
        ).fetchall()
        report_rows = connection.execute(
            "SELECT * FROM stored_reports WHERE report_id != ?", (report_id,)
        ).fetchall()

    exact_by_report = {
        str(row["report_id"]): int(row["matches"]) for row in exact_rows
    }
    core_by_report = {
        str(row["report_id"]): int(row["matches"]) for row in core_rows
    }
    current_fingerprints = int(current_counts["fingerprints"] or 0)
    current_cores = int(current_counts["core_keys"] or 0)
    similarities = []
    for report in report_rows:
        other_id = str(report["report_id"])
        exact_matches = exact_by_report.get(other_id, 0)
        core_matches = core_by_report.get(other_id, 0)
        if not exact_matches and not core_matches:
            continue
        similarities.append(
            {
                "report_id": other_id,
                "report": str(report["report_name"]),
                "filename": str(report["filename"]),
                "exact_findings": exact_matches,
                "exact_overlap": (
                    exact_matches / current_fingerprints
                    if current_fingerprints
                    else 0.0
                ),
                "shared_correction_pairs": core_matches,
                "correction_pair_overlap": (
                    core_matches / current_cores if current_cores else 0.0
                ),
                "findings": int(report["finding_count"]),
                "last_uploaded": str(report["last_uploaded_at"]),
            }
        )
    return sorted(
        similarities,
        key=lambda item: (
            -float(item["exact_overlap"]),
            -int(item["exact_findings"]),
            -int(item["shared_correction_pairs"]),
            str(item["report"]),
        ),
    )[: max(1, int(limit))]


def library_stats(path: str | Path) -> dict[str, int]:
    with closing(_connect(path)) as connection:
        reports = connection.execute(
            "SELECT COUNT(*) AS count FROM stored_reports"
        ).fetchone()["count"]
        findings = connection.execute(
            "SELECT COUNT(*) AS count FROM stored_findings"
        ).fetchone()["count"]
        recurring = connection.execute(
            """
            SELECT COUNT(*) AS count FROM (
                SELECT fingerprint
                FROM stored_findings
                WHERE fingerprint != ''
                GROUP BY fingerprint
                HAVING COUNT(DISTINCT report_id) > 1
            )
            """
        ).fetchone()["count"]
        drafts = connection.execute(
            "SELECT COUNT(*) AS count FROM draft_reviews"
        ).fetchone()["count"]
    return {
        "reports": int(reports),
        "findings": int(findings),
        "recurring_findings": int(recurring),
        "draft_reviews": int(drafts),
    }


def shared_pattern_evidence(path: str | Path) -> list[dict[str, Any]]:
    with closing(_connect(path)) as connection:
        observations = connection.execute(
            "SELECT * FROM observations ORDER BY updated_at"
        ).fetchall()
        core_keys = sorted({str(row["core_key"]) for row in observations})
        stored_rows: list[sqlite3.Row] = []
        for start in range(0, len(core_keys), 800):
            batch = core_keys[start : start + 800]
            placeholders = ",".join("?" for _ in batch)
            stored_rows.extend(
                connection.execute(
                    f"""
                    SELECT f.*, r.report_name
                    FROM stored_findings f
                    JOIN stored_reports r ON r.report_id = f.report_id
                    WHERE f.core_key IN ({placeholders})
                    """,
                    batch,
                ).fetchall()
            )

    observations_by_core: dict[str, list[sqlite3.Row]] = defaultdict(list)
    findings_by_core: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for observation in observations:
        observations_by_core[str(observation["core_key"])].append(observation)
    for finding in stored_rows:
        findings_by_core[str(finding["core_key"])].append(finding)

    evidence = []
    for core_key, human_rows in observations_by_core.items():
        consensus = _consensus(human_rows)
        if not consensus:
            continue
        occurrence_rows = findings_by_core.get(core_key, [])
        representative = occurrence_rows[0] if occurrence_rows else human_rows[0]
        human_report_ids = {str(row["source_report_id"]) for row in human_rows}
        occurrence_report_ids = {str(row["report_id"]) for row in occurrence_rows}
        evidence.append(
            {
                "core_key": core_key,
                "checker": str(representative["checker"]),
                "field": str(representative["field_name"]),
                "original": str(representative["original"]),
                "proposed": str(representative["proposed"]),
                "status": str(consensus["decision"]),
                "comment": str(consensus["review_note"]),
                "human_decisions": len(human_rows),
                "human_reports": len(human_report_ids),
                "occurrences": len(occurrence_rows),
                "course_coverage": len(human_report_ids | occurrence_report_ids),
                "reviewed_sources": str(consensus["memory_source"]),
                "seen_in_reports": ", ".join(
                    sorted({str(row["report_name"]) for row in occurrence_rows})
                ),
            }
        )
    return evidence


def memory_stats(path: str | Path) -> dict[str, int]:
    with closing(_connect(path)) as connection:
        observations = connection.execute(
            "SELECT COUNT(*) AS count FROM observations"
        ).fetchone()["count"]
        findings = connection.execute(
            "SELECT COUNT(DISTINCT fingerprint) AS count FROM observations"
        ).fetchone()["count"]
        reports = connection.execute(
            "SELECT COUNT(DISTINCT source_report_id) AS count FROM observations"
        ).fetchone()["count"]
        conflicts = connection.execute(
            """
            SELECT COUNT(*) AS count FROM (
                SELECT fingerprint
                FROM observations
                GROUP BY fingerprint
                HAVING COUNT(DISTINCT status) > 1
            )
            """
        ).fetchone()["count"]
    return {
        "observations": int(observations),
        "findings": int(findings),
        "reports": int(reports),
        "conflicts": int(conflicts),
    }


def export_memory_bytes(path: str | Path) -> bytes:
    with closing(_connect(path)) as connection:
        observations = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM observations ORDER BY updated_at, observation_id"
            ).fetchall()
        ]
        reports = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM stored_reports ORDER BY first_uploaded_at, report_id"
            ).fetchall()
        ]
        report_findings = [
            dict(row)
            for row in connection.execute(
                """
                SELECT *
                FROM stored_findings
                ORDER BY report_id, finding_order
                """
            ).fetchall()
        ]
        draft_reviews = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM draft_reviews ORDER BY report_id, issue_id"
            ).fetchall()
        ]
        jira_links = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM report_jira_links ORDER BY report_id"
            ).fetchall()
        ]
        review_events = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM review_events ORDER BY occurred_at, event_id"
            ).fetchall()
        ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint_version": FINGERPRINT_VERSION,
        "exported_at": _utc_now(),
        "observations": observations,
        "reports": reports,
        "report_findings": report_findings,
        "draft_reviews": draft_reviews,
        "jira_links": jira_links,
        "review_events": review_events,
    }
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def import_memory_bytes(path: str | Path, payload: bytes) -> dict[str, int]:
    try:
        data = json.load(io.StringIO(payload.decode("utf-8-sig")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DecisionMemoryError("The decision-memory file is not valid JSON.") from error
    imported_schema = data.get("schema_version")
    if imported_schema not in {1, 2, SCHEMA_VERSION}:
        raise DecisionMemoryError("Unsupported decision-memory schema.")
    if data.get("fingerprint_version") != FINGERPRINT_VERSION:
        raise DecisionMemoryError("Unsupported decision fingerprint version.")
    observations = data.get("observations")
    if not isinstance(observations, list):
        raise DecisionMemoryError("observations must be a list.")
    reports = data.get("reports", [])
    report_findings = data.get("report_findings", [])
    draft_reviews = data.get("draft_reviews", [])
    jira_links = data.get("jira_links", [])
    review_events = data.get("review_events", [])
    if not all(
        isinstance(collection, list)
        for collection in (
            reports,
            report_findings,
            draft_reviews,
            jira_links,
            review_events,
        )
    ):
        raise DecisionMemoryError("Report-library collections must be lists.")

    required = set(OBSERVATION_FIELDS)
    imported = 0
    imported_reports = 0
    imported_findings = 0
    imported_drafts = 0
    imported_events = 0
    with closing(_connect(path)) as connection, connection:
        for report in reports:
            if not isinstance(report, dict) or not set(REPORT_FIELDS).issubset(report):
                raise DecisionMemoryError(
                    "A report-library entry is missing required fields."
                )
            connection.execute(
                """
                INSERT INTO stored_reports(
                    report_id, report_name, filename, rules_version,
                    finding_count, reviewable_count, first_uploaded_at,
                    last_uploaded_at, upload_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(report_id) DO UPDATE SET
                    report_name=excluded.report_name,
                    filename=excluded.filename,
                    rules_version=excluded.rules_version,
                    finding_count=excluded.finding_count,
                    reviewable_count=excluded.reviewable_count,
                    first_uploaded_at=MIN(
                        stored_reports.first_uploaded_at,
                        excluded.first_uploaded_at
                    ),
                    last_uploaded_at=MAX(
                        stored_reports.last_uploaded_at,
                        excluded.last_uploaded_at
                    ),
                    upload_count=MAX(
                        stored_reports.upload_count,
                        excluded.upload_count
                    )
                """,
                tuple(report[field] for field in REPORT_FIELDS),
            )
            imported_reports += 1

        for finding in report_findings:
            if (
                not isinstance(finding, dict)
                or not set(REPORT_FINDING_FIELDS).issubset(finding)
            ):
                raise DecisionMemoryError(
                    "A stored finding is missing required fields."
                )
            connection.execute(
                """
                INSERT INTO stored_findings(
                    finding_id, report_id, finding_order, issue_id, fingerprint,
                    core_key, anchor_kind, anchor, checker, field_name, original,
                    proposed, initial_status, initial_comment, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(finding_id) DO UPDATE SET
                    report_id=excluded.report_id,
                    finding_order=excluded.finding_order,
                    issue_id=excluded.issue_id,
                    fingerprint=excluded.fingerprint,
                    core_key=excluded.core_key,
                    anchor_kind=excluded.anchor_kind,
                    anchor=excluded.anchor,
                    checker=excluded.checker,
                    field_name=excluded.field_name,
                    original=excluded.original,
                    proposed=excluded.proposed,
                    initial_status=excluded.initial_status,
                    initial_comment=excluded.initial_comment,
                    record_json=excluded.record_json
                """,
                tuple(finding[field] for field in REPORT_FINDING_FIELDS),
            )
            imported_findings += 1

        for draft in draft_reviews:
            if (
                not isinstance(draft, dict)
                or not set(DRAFT_REVIEW_FIELDS).issubset(draft)
            ):
                raise DecisionMemoryError(
                    "A saved draft review is missing required fields."
                )
            if draft["decision"] not in VALID_STATUSES:
                raise DecisionMemoryError("A draft review has an invalid decision.")
            connection.execute(
                """
                INSERT INTO draft_reviews(
                    report_id, issue_id, decision, note, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(report_id, issue_id) DO UPDATE SET
                    decision=excluded.decision,
                    note=excluded.note,
                    updated_at=MAX(
                        draft_reviews.updated_at,
                        excluded.updated_at
                    )
                """,
                tuple(draft[field] for field in DRAFT_REVIEW_FIELDS),
            )
            imported_drafts += 1

        for jira_link in jira_links:
            if (
                not isinstance(jira_link, dict)
                or not set(JIRA_LINK_FIELDS).issubset(jira_link)
            ):
                raise DecisionMemoryError(
                    "A saved Jira report link is missing required fields."
                )
            connection.execute(
                """
                INSERT INTO report_jira_links(
                    report_id, issue_key, issue_summary, issue_url,
                    attachment_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(report_id) DO UPDATE SET
                    issue_key=excluded.issue_key,
                    issue_summary=excluded.issue_summary,
                    issue_url=excluded.issue_url,
                    attachment_id=excluded.attachment_id,
                    updated_at=MAX(
                        report_jira_links.updated_at,
                        excluded.updated_at
                    )
                """,
                tuple(jira_link[field] for field in JIRA_LINK_FIELDS),
            )

        for event in review_events:
            if (
                not isinstance(event, dict)
                or not set(REVIEW_EVENT_FIELDS).issubset(event)
            ):
                raise DecisionMemoryError(
                    "A reviewer activity entry is missing required fields."
                )
            connection.execute(
                """
                INSERT INTO review_events(
                    event_id, report_id, issue_id, action, decision,
                    reviewer_email, reviewer_name, detail_json, occurred_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO NOTHING
                """,
                tuple(event[field] for field in REVIEW_EVENT_FIELDS),
            )
            imported_events += 1

        for observation in observations:
            if not isinstance(observation, dict) or not required.issubset(observation):
                raise DecisionMemoryError(
                    "A decision-memory observation is missing required fields."
                )
            if observation["status"] not in VALID_STATUSES:
                raise DecisionMemoryError("An observation has an invalid status.")
            connection.execute(
                """
                INSERT INTO observations(
                    observation_id, fingerprint, core_key, anchor_kind, anchor,
                    checker, field_name, original, proposed, status, note,
                    source_report, source_report_id, issue_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(observation_id) DO UPDATE SET
                    fingerprint=excluded.fingerprint,
                    core_key=excluded.core_key,
                    anchor_kind=excluded.anchor_kind,
                    anchor=excluded.anchor,
                    checker=excluded.checker,
                    field_name=excluded.field_name,
                    original=excluded.original,
                    proposed=excluded.proposed,
                    status=excluded.status,
                    note=excluded.note,
                    source_report=excluded.source_report,
                    source_report_id=excluded.source_report_id,
                    issue_id=excluded.issue_id,
                    updated_at=excluded.updated_at
                """,
                tuple(observation[field] for field in OBSERVATION_FIELDS),
            )
            imported += 1
    return {
        "imported": imported,
        "reports": imported_reports,
        "findings": imported_findings,
        "drafts": imported_drafts,
        "events": imported_events,
    }
