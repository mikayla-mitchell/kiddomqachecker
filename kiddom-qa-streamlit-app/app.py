from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import zipfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from auth_access import (
    AccessDeniedError,
    UserIdentity,
    build_user_identity,
    local_development_identity,
)
from decision_memory import (
    DecisionMemoryError,
    export_memory_bytes,
    import_memory_bytes,
    initialize_memory,
    library_stats,
    list_report_review_activity,
    list_report_library,
    load_draft_reviews,
    load_report_jira_link,
    load_report_snapshot,
    match_report_rows,
    memory_stats,
    publish_report_reviews,
    record_review_events,
    report_similarity,
    save_draft_reviews,
    save_report_jira_link,
    shared_pattern_evidence,
    store_report_snapshot,
)
from jira_integration import (
    JiraClient,
    JiraConfig,
    JiraHandoffError,
    JiraIntegrationError,
    is_html_attachment,
)
from pattern_learning import (
    build_shared_pattern_suggestions,
    promote_suggestions,
)
from qa_engine import (
    QAEngineError,
    apply_reviews,
    classify_records,
    detailed_csv_bytes,
    final_csv_bytes,
    finalize_training_comments,
    load_rules,
    parse_decision,
    parse_report_html,
    review_csv_bytes,
    review_progress,
    rules_json_bytes,
    safe_report_name,
    status_counts,
)


APP_DIR = Path(__file__).resolve().parent
BASE_RULES_PATH = APP_DIR / "rules" / "base_rules.json"
MEMORY_PATH = Path(
    os.environ.get("KIDDOM_SHARED_MEMORY_PATH")
    or os.environ.get("KIDDOM_DECISION_MEMORY_PATH")
    or str(APP_DIR / "data" / "shared_memory.sqlite3")
).expanduser()
DECISION_OPTIONS = ["", "approved", "rejected", "needs_change"]
JIRA_REVIEWER_NAMES = ("Karin", "Steve", "Janelle", "Mike")


st.set_page_config(
    page_title="Kiddom QA Review Studio",
    page_icon="✓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --ink: #17324d;
        --muted: #5e7183;
        --paper: #f6f8fb;
        --teal: #1c7c7d;
        --coral: #dc6b4f;
      }
      .stApp { background: linear-gradient(180deg, #fbfcfe 0%, #f5f8fb 100%); }
      [data-testid="stSidebar"] { background: #edf3f6; }
      .qa-kicker {
        color: var(--teal);
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
      }
      .qa-title {
        color: var(--ink);
        font-size: clamp(2rem, 4vw, 3.6rem);
        font-weight: 780;
        letter-spacing: -.045em;
        line-height: 1.02;
        margin: .25rem 0 .55rem;
      }
      .qa-subtitle {
        color: var(--muted);
        font-size: 1.05rem;
        max-width: 860px;
        margin-bottom: 1.2rem;
      }
      div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dfe7ed;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 4px 18px rgba(23, 50, 77, .045);
      }
      .qa-callout {
        background: #eef8f7;
        border-left: 4px solid var(--teal);
        border-radius: 8px;
        color: var(--ink);
        padding: .85rem 1rem;
        margin: .25rem 0 1rem;
      }
      .qa-rule-note {
        color: var(--muted);
        font-size: .9rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def process_payload(
    filename: str, payload: bytes, serialized_rules: bytes
) -> tuple[list[dict], list[dict]]:
    html_text = payload.decode("utf-8", errors="ignore")
    records = parse_report_html(html_text)
    rows = classify_records(records, load_rules(serialized_rules))
    return records, rows


def report_content_id(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _secret_section(name: str) -> Mapping[str, object]:
    try:
        section = st.secrets.get(name, {})
    except (FileNotFoundError, KeyError):
        return {}
    return section if isinstance(section, Mapping) else {}


def access_configuration() -> dict[str, object]:
    section = _secret_section("access")
    domains = section.get("allowed_email_domains", [])
    admins = section.get("admin_emails", [])
    return {
        "allowed_email_domains": domains,
        "admin_emails": admins,
        "jira_reviewer_mode": str(
            section.get("jira_reviewer_mode") or "self"
        ).strip().casefold(),
    }


def google_auth_is_configured() -> bool:
    return bool(_secret_section("auth"))


def enforce_google_workspace_login() -> UserIdentity:
    if not google_auth_is_configured():
        return local_development_identity()

    if not st.user.is_logged_in:
        st.markdown(
            '<div class="qa-kicker">Kiddom Google Workspace</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="qa-title">Sign in to QA Review Studio</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="qa-subtitle">
              Use your work Google account. Your identity is used to load your
              Jira assignments and record reviewer activity outside the final
              training CSV.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Continue with Google", type="primary"):
            st.login()
        st.stop()

    claims = st.user.to_dict()
    access = access_configuration()
    try:
        return build_user_identity(
            claims,
            allowed_email_domains=access["allowed_email_domains"],
            admin_emails=access["admin_emails"],
        )
    except AccessDeniedError as error:
        st.error(str(error))
        if st.button("Sign out"):
            st.logout()
        st.stop()


def jira_account_mapping() -> dict[str, str]:
    section = _secret_section("jira_user_map")
    return {
        str(email).strip().casefold(): str(account_id).strip()
        for email, account_id in section.items()
        if str(email).strip() and str(account_id).strip()
    }


def jira_reviewer_mapping() -> dict[str, str]:
    section = _secret_section("jira_reviewers")
    return {
        str(label).strip().casefold(): str(account_id).strip()
        for label, account_id in section.items()
        if str(label).strip() and str(account_id).strip()
    }


def initialize_state(user: UserIdentity) -> None:
    if "rules" not in st.session_state:
        st.session_state.rules = load_rules(BASE_RULES_PATH)
    if "reports" not in st.session_state:
        st.session_state.reports = {}
    if "selected_report" not in st.session_state:
        st.session_state.selected_report = None
    if "upload_generation" not in st.session_state:
        st.session_state.upload_generation = 0
    if "jira_people" not in st.session_state:
        st.session_state.jira_people = []
    if "jira_issues" not in st.session_state:
        st.session_state.jira_issues = []
    if "jira_qa_people" not in st.session_state:
        st.session_state.jira_qa_people = []
    if "jira_admin_people" not in st.session_state:
        st.session_state.jira_admin_people = []
    if "jira_reviewer_directory" not in st.session_state:
        st.session_state.jira_reviewer_directory = {}
    if st.session_state.get("active_reviewer_email") != user.email:
        st.session_state.jira_people = []
        st.session_state.jira_issues = []
        st.session_state.jira_reviewer_directory = {}
        st.session_state.pop("jira_loaded_reviewer_id", None)
        st.session_state.pop("jira_loaded_reviewer_label", None)
        st.session_state.pop("jira_selected_issue", None)
        st.session_state.active_reviewer_email = user.email
    if "memory_initialized" not in st.session_state:
        try:
            initialize_memory(MEMORY_PATH)
            st.session_state.memory_error = ""
        except (DecisionMemoryError, OSError) as error:
            st.session_state.memory_error = str(error)
        st.session_state.memory_initialized = True


def record_report_activity(
    report: dict,
    action: str,
    events: list[dict],
    user: UserIdentity,
) -> None:
    if (
        not events
        or st.session_state.memory_error
        or not report.get("library_saved")
    ):
        return
    try:
        record_review_events(
            MEMORY_PATH,
            report["report_id"],
            action,
            user.as_dict(),
            events,
        )
    except (DecisionMemoryError, OSError) as error:
        st.warning(f"The review was saved, but its reviewer audit was not: {error}")


def normalize_ui_decision(raw: str, note: str = "") -> tuple[str, str]:
    status, parsed_note = parse_decision(raw, note)
    return status or "", parsed_note or ""


def import_review_sheet(payload: bytes, report: dict) -> tuple[int, int]:
    text = payload.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames or []
    if "issue_id" not in fields:
        raise QAEngineError("The review CSV must contain an issue_id column.")
    decision_column = fields[0] if fields[0] != "issue_id" else "decision"
    valid_ids = {
        str(row["issue_id"])
        for row in report["rows"]
        if row["status"] == "needs_change"
    }
    applied = 0
    unmatched = 0
    for row in reader:
        issue_id = str(row.get("issue_id") or "")
        if issue_id not in valid_ids:
            unmatched += 1
            continue
        decision, note = normalize_ui_decision(
            str(row.get(decision_column) or ""),
            str(row.get("review_note") or row.get("comment") or ""),
        )
        if not decision:
            continue
        report["reviews"][issue_id] = {
            "decision": decision,
            "review_note": note,
        }
        applied += 1
    return applied, unmatched


def make_review_frame(report: dict) -> pd.DataFrame:
    rows = []
    for row in report["rows"]:
        if row["status"] != "needs_change":
            continue
        review = report["reviews"].get(str(row["issue_id"]), {})
        context = (
            f"{row.get('context_before', '')}  ‹CHANGE›  "
            f"{row.get('context_after', '')}"
        ).strip()
        rows.append(
            {
                "decision": review.get("decision", ""),
                "review_note": review.get("review_note", ""),
                "checker": row["checker"],
                "field": row["field"],
                "original": row["original"],
                "proposed": row["proposed"],
                "context": context,
                "location": row["breadcrumb"] or row["node_label"],
                "reused_from": review.get("memory_source", ""),
                "match_quality": (
                    f"{review.get('memory_match', '').title()} "
                    f"{float(review.get('memory_score', 0)):.0%}"
                    if review.get("memory_match")
                    else ""
                ),
                "node_link": row["node_link"],
                "issue_id": row["issue_id"],
            }
        )
    return pd.DataFrame(rows)


def build_export_zip(report: dict) -> bytes:
    merged = apply_reviews(report["rows"], report["reviews"])
    prefix = report["name"]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{prefix}_FINAL.csv", final_csv_bytes(merged))
        archive.writestr(
            f"{prefix}_flagged_for_review.csv",
            review_csv_bytes(report["rows"], report["reviews"]),
        )
        archive.writestr(
            f"{prefix}_detailed.csv", detailed_csv_bytes(report["rows"])
        )
        archive.writestr("rules_used.json", rules_json_bytes(st.session_state.rules))
    return stream.getvalue()


def reclassify_loaded_reports() -> None:
    for report in st.session_state.reports.values():
        report["rows"] = classify_records(report["records"], st.session_state.rules)


def seed_reviews_from_memory(report: dict) -> dict[str, int]:
    if st.session_state.memory_error:
        return {"reused": 0, "exact": 0, "near": 0}
    matches = match_report_rows(MEMORY_PATH, report["rows"])
    reused = 0
    exact = 0
    near = 0
    for issue_id, match in matches.items():
        existing = report["reviews"].get(issue_id, {})
        if str(existing.get("decision") or "").strip():
            continue
        report["reviews"][issue_id] = dict(match)
        reused += 1
        exact += int(match.get("memory_match") == "exact")
        near += int(match.get("memory_match") == "near")
    report["memory_reused"] = sum(
        bool(review.get("memory_match")) for review in report["reviews"].values()
    )
    return {"reused": reused, "exact": exact, "near": near}


def seed_all_loaded_reports() -> dict[str, int]:
    total = {"reused": 0, "exact": 0, "near": 0}
    for loaded_report in st.session_state.reports.values():
        result = seed_reviews_from_memory(loaded_report)
        for key in total:
            total[key] += result[key]
    return total


def open_saved_report(report_id: str) -> dict[str, int]:
    snapshot = load_report_snapshot(MEMORY_PATH, report_id)
    records = snapshot["records"]
    rows = classify_records(records, st.session_state.rules)
    saved_drafts = load_draft_reviews(MEMORY_PATH, report_id)
    reviewable_ids = {
        str(row["issue_id"]) for row in rows if row["status"] == "needs_change"
    }
    reviews = {
        issue_id: saved_drafts.get(
            issue_id, {"decision": "", "review_note": ""}
        )
        for issue_id in reviewable_ids
    }
    loaded_report = {
        "report_id": report_id,
        "name": snapshot["name"],
        "filename": snapshot["filename"],
        "records": records,
        "rows": rows,
        "reviews": reviews,
        "memory_reused": 0,
        "from_library": True,
        "library_saved": True,
    }
    jira_link = load_report_jira_link(MEMORY_PATH, report_id)
    if jira_link:
        loaded_report["jira_issue"] = jira_link
        loaded_report["jira_attachment_id"] = jira_link.get("attachment_id", "")
    st.session_state.reports[report_id] = loaded_report
    st.session_state.selected_report = report_id
    return seed_reviews_from_memory(loaded_report)


def add_report_payload(
    filename: str,
    payload: bytes,
    source_metadata: dict | None = None,
) -> dict:
    """Run the standard report pipeline for uploads and Jira attachments."""
    report_id = report_content_id(payload)
    records, rows = process_payload(
        filename,
        payload,
        rules_json_bytes(st.session_state.rules),
    )
    report_name = safe_report_name(filename)
    library_saved = False
    library_error = ""
    stored_findings = 0
    if not st.session_state.memory_error:
        try:
            stored = store_report_snapshot(
                MEMORY_PATH,
                report_id,
                report_name,
                filename,
                records,
                rows,
                st.session_state.rules["rules_version"],
            )
            saved_drafts = load_draft_reviews(MEMORY_PATH, report_id)
            library_saved = True
            stored_findings = stored["stored"]
        except (DecisionMemoryError, OSError) as error:
            library_error = str(error)
            saved_drafts = {}
    else:
        saved_drafts = {}
    old_reviews = st.session_state.reports.get(report_id, {}).get("reviews", {})
    reviews = {
        str(row["issue_id"]): old_reviews.get(
            str(row["issue_id"]),
            saved_drafts.get(
                str(row["issue_id"]),
                {"decision": "", "review_note": ""},
            ),
        )
        for row in rows
        if row["status"] == "needs_change"
    }
    loaded_report = {
        "report_id": report_id,
        "name": report_name,
        "filename": filename,
        "records": records,
        "rows": rows,
        "reviews": reviews,
        "memory_reused": 0,
        "from_library": False,
        "library_saved": library_saved,
        **(source_metadata or {}),
    }
    jira_issue = loaded_report.get("jira_issue")
    if library_saved and jira_issue:
        try:
            save_report_jira_link(
                MEMORY_PATH,
                report_id,
                jira_issue,
                str(loaded_report.get("jira_attachment_id") or ""),
            )
        except (DecisionMemoryError, OSError) as error:
            library_error = (
                f"{library_error}; " if library_error else ""
            ) + f"Jira link was not saved: {error}"
    st.session_state.reports[report_id] = loaded_report
    memory_result = seed_reviews_from_memory(loaded_report)
    st.session_state.selected_report = report_id
    return {
        "report": loaded_report,
        "memory": memory_result,
        "stored_findings": stored_findings,
        "library_saved": library_saved,
        "library_error": library_error,
    }


JIRA_SETTING_NAMES = (
    "JIRA_BASE_URL",
    "JIRA_USER_EMAIL",
    "JIRA_API_TOKEN",
    "JIRA_PROJECT_KEY",
    "JIRA_READY_FOR_QA_STATUS",
    "JIRA_QA_ACCOUNT_ID",
    "JIRA_TICKET_JQL",
    "JIRA_MAX_RESULTS",
)


def jira_configuration() -> tuple[JiraConfig | None, str]:
    values: dict[str, object] = {
        name: os.environ.get(name, "") for name in JIRA_SETTING_NAMES
    }
    try:
        nested = st.secrets.get("jira", {})
        for name in JIRA_SETTING_NAMES:
            if name in st.secrets:
                values[name] = st.secrets[name]
            elif isinstance(nested, Mapping):
                nested_name = name.removeprefix("JIRA_").lower()
                if nested_name in nested:
                    values[name] = nested[nested_name]
    except (FileNotFoundError, KeyError):
        pass
    try:
        return JiraConfig.from_mapping(values), ""
    except JiraIntegrationError as error:
        return None, str(error)


def jira_client(config: JiraConfig) -> JiraClient:
    return JiraClient(config)


def render_jira_setup(configuration_error: str) -> None:
    st.info(configuration_error)
    st.markdown(
        """
        Configure the deployment with a Jira Cloud service account:

        ```toml
        # .streamlit/secrets.toml
        [jira]
        base_url = "https://your-company.atlassian.net"
        user_email = "jira-service-account@example.com"
        api_token = "..."
        project_key = "CURR"
        ready_for_qa_status = "Ready for QA"
        qa_account_id = "..."
        ```

        The API token stays server-side. Protect the Streamlit deployment with
        your organization's normal access controls.
        """
    )


def jira_reviewer_candidates(
    client: JiraClient,
    label: str,
    *,
    refresh: bool = False,
) -> list[dict[str, str]]:
    directory = st.session_state.jira_reviewer_directory
    cache_key = label.strip().casefold()
    mapped_account_id = jira_reviewer_mapping().get(cache_key)
    if mapped_account_id:
        people = [
            {
                "account_id": mapped_account_id,
                "display_name": label,
                "email": "",
                "active": True,
            }
        ]
        directory[cache_key] = people
        return people
    if refresh or cache_key not in directory:
        directory[cache_key] = client.search_named_users(label)
    return directory[cache_key]


def render_jira_person_choice(
    label: str,
    people: list[dict[str, str]],
    *,
    key: str,
) -> dict[str, str] | None:
    if not people:
        st.error(f'Jira could not find an active user named "{label}".')
        return None
    if len(people) == 1:
        person = people[0]
        st.caption(f"Jira account: {person['display_name']}")
        return person
    people_by_id = {person["account_id"]: person for person in people}
    account_id = st.selectbox(
        f"Jira account for {label}",
        options=list(people_by_id),
        format_func=lambda candidate_id: (
            people_by_id[candidate_id]["display_name"]
            + (
                f" · {people_by_id[candidate_id]['email']}"
                if people_by_id[candidate_id]["email"]
                else ""
            )
        ),
        key=key,
        help=(
            f'Jira found more than one active user matching "{label}". '
            "Choose the intended account before continuing."
        ),
    )
    return people_by_id[account_id]


def store_refreshed_jira_issue(issue: dict) -> None:
    st.session_state.jira_issues = [
        issue if current["key"] == issue["key"] else current
        for current in st.session_state.jira_issues
    ]
    st.session_state.jira_selected_issue = issue


def jira_link_label(link: str, index: int) -> str:
    parsed = urlparse(link)
    path = parsed.path.rstrip("/")
    detail = path.rsplit("/", 1)[-1] if path else ""
    label = parsed.netloc.removeprefix("www.")
    if detail and len(detail) <= 44:
        label += f" · {detail}"
    return label or f"Ticket link {index}"


def render_jira_workspace(
    config: JiraConfig | None,
    configuration_error: str,
    user: UserIdentity,
) -> None:
    del user
    st.markdown("### Jira ticket workspace")
    st.markdown(
        """
        Choose a reviewer to see their open Jira tickets. Open ticket links,
        change status, reassign work, or load an attached Issue Annotation
        Report without leaving this workspace.
        """
    )
    if config is None:
        render_jira_setup(configuration_error)
        return
    client = jira_client(config)

    st.caption(
        f"Ticket scope: project {config.project_key}"
        if config.project_key
        else "Ticket scope: all visible Jira projects"
    )

    reviewer_col, refresh_col = st.columns([3, 1])
    with reviewer_col:
        reviewer_label = st.selectbox(
            "Whose tickets do you want to open?",
            options=JIRA_REVIEWER_NAMES,
            index=None,
            placeholder="Choose Karin, Steve, Janelle, or Mike",
            key="jira-directory-reviewer",
        )
    with refresh_col:
        st.write("")
        refresh_clicked = st.button(
            "Refresh tickets",
            type="primary",
            width="stretch",
            disabled=reviewer_label is None,
            key="jira-refresh-directory",
        )

    if reviewer_label is None:
        st.info("Choose a reviewer to load their assigned Jira tickets.")
        return

    try:
        reviewer_people = jira_reviewer_candidates(
            client,
            reviewer_label,
            refresh=refresh_clicked,
        )
    except JiraIntegrationError as error:
        st.error(str(error))
        return
    reviewer = render_jira_person_choice(
        reviewer_label,
        reviewer_people,
        key=f"jira-reviewer-account-{reviewer_label.casefold()}",
    )
    if reviewer is None:
        return

    reviewer_changed = (
        st.session_state.get("jira_loaded_reviewer_id")
        != reviewer["account_id"]
    )
    if reviewer_changed or refresh_clicked:
        try:
            with st.spinner(
                f"Loading Jira tickets for {reviewer['display_name']}…"
            ):
                st.session_state.jira_issues = client.search_assigned_issues(
                    reviewer["account_id"]
                )
            st.session_state.jira_people = [reviewer]
            st.session_state.jira_loaded_reviewer_id = reviewer["account_id"]
            st.session_state.jira_loaded_reviewer_label = reviewer_label
            if reviewer_changed:
                st.session_state.pop("jira_selected_issue", None)
        except JiraIntegrationError as error:
            st.session_state.jira_issues = []
            st.error(str(error))
            return

    issues = st.session_state.jira_issues
    if not issues:
        st.info(f"No open {config.project_key or 'Jira'} tickets are assigned to "
                f"{reviewer['display_name']}.")
        return

    issues_by_key = {issue["key"]: issue for issue in issues}
    issue_key = st.selectbox(
        "Assigned ticket",
        options=list(issues_by_key),
        format_func=lambda key: (
            f"{key} · {issues_by_key[key]['summary']} "
            f"· {issues_by_key[key]['status']}"
        ),
        key=f"jira-issue-{reviewer['account_id']}",
    )
    issue = issues_by_key[issue_key]
    st.session_state.jira_selected_issue = issue

    flash = st.session_state.pop("jira_action_flash", None)
    if flash:
        st.success(flash)

    ticket_col, status_col, assignee_col, refresh_ticket_col = st.columns(
        [1.3, 1, 1, 1]
    )
    ticket_col.link_button(
        f"Open {issue['key']} in Jira",
        issue["browse_url"],
        width="stretch",
    )
    status_col.metric("Status", issue["status"] or "Unknown")
    assignee_col.metric("Assignee", issue["assignee_name"] or "Unassigned")
    if refresh_ticket_col.button(
        "Refresh ticket",
        width="stretch",
        key=f"jira-refresh-ticket-{issue['key']}",
    ):
        try:
            refreshed_issue = client.get_issue(issue["key"])
            store_refreshed_jira_issue(refreshed_issue)
            st.session_state.jira_action_flash = (
                f"Refreshed {refreshed_issue['key']} from Jira."
            )
            st.rerun()
        except JiraIntegrationError as error:
            st.error(str(error))

    st.markdown(f"**{issue['summary']}**")
    if issue["updated"]:
        st.caption(f"Last updated in Jira: {issue['updated']}")

    st.markdown("#### Ticket actions")
    action_status_col, action_assignee_col = st.columns(2)
    with action_status_col:
        try:
            transitions = client.get_transitions(issue["key"])
        except JiraIntegrationError as error:
            transitions = []
            st.error(str(error))
        transitions_by_id = {
            str(item.get("id") or ""): item
            for item in transitions
            if item.get("id")
        }
        transition_id = st.selectbox(
            "Change status",
            options=list(transitions_by_id),
            index=None,
            placeholder=(
                "Choose an available Jira status"
                if transitions_by_id
                else "No status changes are available"
            ),
            format_func=lambda item_id: (
                str(transitions_by_id[item_id].get("name") or "Move")
                + " → "
                + str(
                    (transitions_by_id[item_id].get("to") or {}).get("name")
                    or "Unknown"
                )
            ),
            disabled=not transitions_by_id,
            key=f"jira-transition-{issue['key']}",
        )
        if st.button(
            "Update status",
            type="primary",
            width="stretch",
            disabled=transition_id is None,
            key=f"jira-update-status-{issue['key']}",
        ):
            try:
                new_status = client.transition_issue_by_id(
                    issue["key"],
                    transition_id,
                )
                store_refreshed_jira_issue(client.get_issue(issue["key"]))
                st.session_state.jira_action_flash = (
                    f"Moved {issue['key']} to {new_status}."
                )
                st.rerun()
            except JiraIntegrationError as error:
                st.error(str(error))

    with action_assignee_col:
        target_label = st.selectbox(
            "Reassign to",
            options=JIRA_REVIEWER_NAMES,
            index=None,
            placeholder="Choose Karin, Steve, Janelle, or Mike",
            key=f"jira-target-label-{issue['key']}",
        )
        target_person = None
        if target_label:
            try:
                target_people = jira_reviewer_candidates(client, target_label)
                target_person = render_jira_person_choice(
                    target_label,
                    target_people,
                    key=(
                        f"jira-target-account-{issue['key']}-"
                        f"{target_label.casefold()}"
                    ),
                )
            except JiraIntegrationError as error:
                st.error(str(error))
        already_assigned = bool(
            target_person
            and target_person["account_id"] == issue["assignee_account_id"]
        )
        if already_assigned:
            st.caption(f"{issue['key']} is already assigned to this person.")
        if st.button(
            "Reassign ticket",
            width="stretch",
            disabled=target_person is None or already_assigned,
            key=f"jira-reassign-{issue['key']}",
        ):
            try:
                client.assign_issue(issue["key"], target_person["account_id"])
                store_refreshed_jira_issue(client.get_issue(issue["key"]))
                st.session_state.jira_action_flash = (
                    f"Reassigned {issue['key']} to "
                    f"{target_person['display_name']}."
                )
                st.rerun()
            except JiraIntegrationError as error:
                st.error(str(error))

    if issue["description"]:
        with st.expander("Ticket description"):
            st.write(issue["description"])

    html_attachments = [
        attachment
        for attachment in issue["attachments"]
        if is_html_attachment(attachment)
    ]
    st.markdown("#### HTML reports")
    if not html_attachments:
        st.info(
            "This ticket has no HTML attachment. Open any ticket links below "
            "to download the report, then use the normal HTML uploader."
        )
    else:
        attachments_by_id = {
            attachment["id"]: attachment for attachment in html_attachments
        }
        attachment_id = st.selectbox(
            "Issue Annotation Report attachment",
            options=list(attachments_by_id),
            format_func=lambda item_id: (
                f"{attachments_by_id[item_id]['filename']} "
                f"({attachments_by_id[item_id]['size'] / (1024 * 1024):.1f} MB)"
            ),
            key="jira-html-attachment",
        )
        attachment = attachments_by_id[attachment_id]
        if st.button(
            "Load HTML into QA review",
            type="primary",
            key="jira-load-html",
        ):
            try:
                with st.spinner(
                    f"Downloading and parsing {attachment['filename']}…"
                ):
                    payload = client.download_attachment(attachment["id"])
                    result = add_report_payload(
                        attachment["filename"],
                        payload,
                        {
                            "jira_issue": issue,
                            "jira_attachment_id": attachment["id"],
                        },
                    )
                del payload
                st.session_state.processing_flash = {
                    **result["memory"],
                    "reports": int(result["library_saved"]),
                    "findings": result["stored_findings"],
                }
                if result["library_error"]:
                    st.session_state.jira_load_warning = result["library_error"]
                st.session_state.requested_primary_workspace = "Review workspace"
                st.rerun()
            except (JiraIntegrationError, QAEngineError, UnicodeError, ValueError) as error:
                st.error(str(error))

    if st.session_state.reports:
        active_report_id = st.session_state.selected_report
        active_report = st.session_state.reports.get(active_report_id)
        if active_report and st.button(
            "Link active report to this ticket",
            key="jira-link-active-report",
        ):
            active_report["jira_issue"] = issue
            if active_report.get("library_saved") and not st.session_state.memory_error:
                try:
                    save_report_jira_link(
                        MEMORY_PATH,
                        active_report["report_id"],
                        issue,
                        str(active_report.get("jira_attachment_id") or ""),
                    )
                except (DecisionMemoryError, OSError) as error:
                    st.warning(f"The report is linked in this session only: {error}")
            st.success(
                f"Linked {active_report['filename']} to Jira ticket {issue['key']}."
            )

    if issue["links"]:
        st.markdown("#### Links from the ticket")
        for index, link in enumerate(issue["links"], start=1):
            st.link_button(
                jira_link_label(link, index),
                link,
                width="stretch",
            )


def render_jira_handoff(
    report: dict,
    merged_rows: list[dict],
    outstanding: int,
    config: JiraConfig | None,
    configuration_error: str,
    user: UserIdentity,
) -> None:
    st.divider()
    st.markdown("#### Send completed review to Jira")
    issue = report.get("jira_issue")
    selected_issue = st.session_state.get("jira_selected_issue")
    if issue is None and selected_issue:
        st.caption(
            f"Most recently selected ticket: {selected_issue['key']} · "
            f"{selected_issue['summary']}"
        )
        if st.button(
            "Link this report to the selected ticket",
            key=f"jira-associate-{report['report_id']}",
        ):
            report["jira_issue"] = selected_issue
            if report.get("library_saved") and not st.session_state.memory_error:
                try:
                    save_report_jira_link(
                        MEMORY_PATH,
                        report["report_id"],
                        selected_issue,
                        str(report.get("jira_attachment_id") or ""),
                    )
                except (DecisionMemoryError, OSError) as error:
                    st.warning(f"The report is linked in this session only: {error}")
            st.rerun()
    issue = report.get("jira_issue")
    if issue is None:
        st.info(
            "Load this report from Jira, or link it to a ticket in the Jira "
            "ticket workspace, before sending the completed CSV."
        )
        return
    st.link_button(
        f"Open {issue['key']} · {issue['summary']}",
        issue["browse_url"],
    )
    if config is None:
        render_jira_setup(configuration_error)
        return

    client = jira_client(config)
    qa_account_id = config.qa_account_id
    qa_owner_label = "the configured QA owner"
    if not qa_account_id:
        with st.expander("Choose the QA owner"):
            with st.form(f"jira-qa-search-{report['report_id']}"):
                qa_query = st.text_input(
                    "QA owner name or email",
                    key=f"jira-qa-query-{report['report_id']}",
                )
                qa_search = st.form_submit_button(
                    "Search Jira people",
                    disabled=not qa_query.strip(),
                )
            if qa_search:
                try:
                    st.session_state.jira_qa_people = client.search_users(qa_query)
                except JiraIntegrationError as error:
                    st.error(str(error))
            qa_people = st.session_state.jira_qa_people
            if qa_people:
                qa_by_id = {person["account_id"]: person for person in qa_people}
                qa_account_id = st.selectbox(
                    "Reassign to",
                    options=list(qa_by_id),
                    format_func=lambda account_id: qa_by_id[account_id][
                        "display_name"
                    ],
                    key=f"jira-qa-owner-{report['report_id']}",
                )
                qa_owner_label = qa_by_id[qa_account_id]["display_name"]
    else:
        st.caption(
            "QA reassign target is controlled by the deployment's "
            "`JIRA_QA_ACCOUNT_ID` setting."
        )

    csv_payload = final_csv_bytes(merged_rows)
    content_hash = hashlib.sha256(csv_payload).hexdigest()[:10]
    csv_filename = f"{report['name']}_FINAL_{content_hash}.csv"
    confirmation = st.checkbox(
        (
            f"Attach {csv_filename}, move {issue['key']} to "
            f"{config.ready_for_qa_status}, and reassign it to {qa_owner_label}."
        ),
        key=f"jira-confirm-{report['report_id']}-{issue['key']}",
    )
    disabled_reason = ""
    if outstanding:
        disabled_reason = (
            f"Complete the remaining {outstanding:,} human decisions first."
        )
    elif not qa_account_id:
        disabled_reason = "Choose a QA owner first."
    elif not confirmation:
        disabled_reason = "Confirm the three Jira updates first."
    if disabled_reason:
        st.caption(disabled_reason)

    if st.button(
        "Attach CSV, mark for QA, and reassign",
        type="primary",
        disabled=bool(disabled_reason),
        key=f"jira-handoff-{report['report_id']}-{issue['key']}",
    ):
        try:
            with st.spinner(f"Updating Jira ticket {issue['key']}…"):
                steps = client.handoff_completed_review(
                    issue["key"],
                    csv_filename,
                    csv_payload,
                    qa_account_id,
                    config.ready_for_qa_status,
                )
            report["jira_handoff"] = steps
            record_report_activity(
                report,
                "jira_handoff",
                [
                    {
                        "detail": {
                            "issue_key": issue["key"],
                            "csv_filename": csv_filename,
                            "steps": steps,
                        }
                    }
                ],
                user,
            )
            st.success(f"Jira ticket {issue['key']} is ready for QA.")
            st.dataframe(pd.DataFrame(steps), hide_index=True, width="stretch")
        except JiraHandoffError as error:
            if error.completed_steps:
                st.warning(
                    "Jira accepted some steps before the handoff stopped. "
                    "Retrying is safe: completed steps are detected and skipped."
                )
                st.dataframe(
                    pd.DataFrame(error.completed_steps),
                    hide_index=True,
                    width="stretch",
                )
            st.error(str(error))


current_user = enforce_google_workspace_login()
initialize_state(current_user)
jira_config, jira_config_error = jira_configuration()

st.markdown('<div class="qa-kicker">Curriculum quality operations</div>', unsafe_allow_html=True)
st.markdown('<div class="qa-title">Kiddom QA Review Studio</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="qa-subtitle">
      Open an assigned Jira ticket or drop in Issue Annotation Report HTML
      files, triage findings with the established curriculum-aware rules,
      preserve parsed reports in a shared library, reuse matching decisions
      across course variants, and send training-safe results to QA.
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    if current_user.is_authenticated:
        st.markdown(f"**{current_user.name}**")
        st.caption(current_user.email)
        if current_user.is_admin:
            st.caption("App administrator")
        if st.button("Sign out", width="stretch"):
            st.logout()
        st.divider()
    else:
        st.caption("Local development mode · Google sign-in is not configured")

    st.header("1 · Add reports")
    uploads = st.file_uploader(
        "Kiddom report HTML",
        type=["html", "htm"],
        accept_multiple_files=True,
        key=f"report-uploads-{st.session_state.upload_generation}",
        help="You can process multiple courses in the same session.",
    )
    process_clicked = st.button(
        "Process uploaded reports",
        type="primary",
        width="stretch",
        disabled=not uploads,
    )

    if process_clicked and uploads:
        memory_reuse_total = {"reused": 0, "exact": 0, "near": 0}
        library_total = {"reports": 0, "findings": 0}
        for upload in uploads:
            payload = upload.getvalue()
            with st.spinner(f"Reading {upload.name}…"):
                try:
                    result = add_report_payload(upload.name, payload)
                except (QAEngineError, UnicodeError, ValueError) as error:
                    st.error(f"{upload.name}: {error}")
                    continue
            if result["library_error"]:
                st.warning(
                    f"{upload.name} was processed but could not be saved "
                    f"to the shared library: {result['library_error']}"
                )
            memory_result = result["memory"]
            for result_key in memory_reuse_total:
                memory_reuse_total[result_key] += memory_result[result_key]
            library_total["reports"] += int(result["library_saved"])
            library_total["findings"] += result["stored_findings"]
            del payload
        st.session_state.upload_generation += 1
        st.session_state.processing_flash = {
            **memory_reuse_total,
            **library_total,
        }
        st.rerun()

    st.divider()
    st.header("2 · Jira")
    if jira_config:
        st.success("Jira Cloud is connected to this deployment.")
    else:
        st.caption("Configure Jira to load assigned tickets and return completed CSVs.")
    if st.button("Open Jira ticket workspace", width="stretch"):
        st.session_state.requested_primary_workspace = "Jira tickets"
        st.rerun()

    st.divider()
    st.header("3 · Shared library")
    saved_reports = (
        list_report_library(MEMORY_PATH)
        if not st.session_state.memory_error
        else []
    )
    saved_report_by_id = {
        str(item["report_id"]): item for item in saved_reports
    }
    saved_report_id = st.selectbox(
        "Open a saved report",
        options=list(saved_report_by_id),
        format_func=lambda report_id: (
            f"{saved_report_by_id[report_id]['report_name']} "
            f"({saved_report_by_id[report_id]['finding_count']:,} findings)"
        ),
        disabled=not saved_reports,
        placeholder="No saved reports yet",
    )
    if st.button(
        "Open saved report",
        width="stretch",
        disabled=not saved_report_id,
    ):
        open_saved_report(str(saved_report_id))
        st.rerun()
    if saved_reports:
        st.caption(
            f"{len(saved_reports):,} parsed reports are available to every user."
        )

    st.divider()
    st.header("4 · Rulebook")
    st.caption(f"Active version: {st.session_state.rules['rules_version']}")
    custom_rules = st.file_uploader(
        "Load an updated rulebook",
        type=["json"],
        key="custom-rules-upload",
    )
    if st.button(
        "Use uploaded rulebook",
        width="stretch",
        disabled=custom_rules is None,
    ):
        try:
            st.session_state.rules = load_rules(custom_rules.getvalue())
            reclassify_loaded_reports()
            st.success("Rulebook loaded and reports reclassified.")
            st.rerun()
        except (QAEngineError, ValueError) as error:
            st.error(str(error))
    if st.button("Reset to bundled rules", width="stretch"):
        st.session_state.rules = load_rules(BASE_RULES_PATH)
        reclassify_loaded_reports()
        st.success("Bundled rules restored.")
        st.rerun()

    st.divider()
    st.caption(
        "Raw HTML is discarded after parsing. Compact report snapshots, saved "
        "review drafts, and published decisions are shared by this deployment."
    )
    if st.session_state.memory_error:
        st.error(f"Decision memory unavailable: {st.session_state.memory_error}")

processing_flash = st.session_state.pop("processing_flash", None)
if processing_flash is not None:
    stored_message = (
        f" Saved {processing_flash.get('reports', 0):,} report snapshot"
        f"{'s' if processing_flash.get('reports', 0) != 1 else ''} "
        f"with {processing_flash.get('findings', 0):,} findings to the shared library."
        if processing_flash.get("reports", 0)
        else ""
    )
    if processing_flash["reused"]:
        st.toast(
            "Processing complete — "
            f"{processing_flash['reused']:,} prior decisions reused "
            f"({processing_flash['exact']:,} exact, "
            f"{processing_flash['near']:,} near).{stored_message}",
            icon="♻️",
        )
    else:
        st.toast(
            f"Processing complete. No prior decisions matched.{stored_message}",
            icon="✅",
        )

jira_load_warning = st.session_state.pop("jira_load_warning", None)
if jira_load_warning:
    st.warning(
        "The Jira HTML was processed, but its snapshot could not be saved to "
        f"the shared library: {jira_load_warning}"
    )

requested_workspace = st.session_state.pop("requested_primary_workspace", None)
if requested_workspace:
    st.session_state.primary_workspace = requested_workspace
primary_workspace = st.radio(
    "Primary workspace",
    ["Review workspace", "Jira tickets"],
    horizontal=True,
    label_visibility="collapsed",
    key="primary_workspace",
)
if primary_workspace == "Jira tickets":
    render_jira_workspace(jira_config, jira_config_error, current_user)
    st.stop()


if not st.session_state.reports:
    st.markdown(
        """
        <div class="qa-callout">
          Start in the sidebar. The app accepts the same generated HTML reports
          used by the Codex skill, saves a compact parsed snapshot to the shared
          library, and preserves the exact three-column, fully-quoted Kiddom
          import format.
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, middle, right = st.columns(3)
    left.markdown("#### Parse\nDeduplicates every atomic checker finding.")
    middle.markdown("#### Review\nFocuses humans only on ambiguous decisions.")
    right.markdown(
        "#### Reuse\nCarries confirmed decisions into matching course variants."
    )
    if saved_reports:
        st.markdown("#### Shared report library")
        empty_library_frame = pd.DataFrame(
            [
                {
                    "Report": item["report_name"],
                    "Findings": item["finding_count"],
                    "Saved decisions": item["draft_reviews"],
                    "Published decisions": item["published_reviews"],
                    "Last uploaded": item["last_uploaded_at"],
                }
                for item in saved_reports
            ]
        )
        st.dataframe(empty_library_frame, width="stretch", hide_index=True)
        st.caption("Open any saved report from the sidebar to continue reviewing it.")
    st.stop()


report_options = list(st.session_state.reports)
if st.session_state.selected_report not in report_options:
    st.session_state.selected_report = report_options[0]

selected_key = st.selectbox(
    "Active report",
    options=report_options,
    index=report_options.index(st.session_state.selected_report),
    format_func=lambda key: st.session_state.reports[key]["filename"],
)
st.session_state.selected_report = selected_key
report = st.session_state.reports[selected_key]
if report.get("jira_issue"):
    source_issue = report["jira_issue"]
    st.caption(
        f"Linked Jira ticket: [{source_issue['key']} · "
        f"{source_issue['summary']}]({source_issue['browse_url']})"
    )
merged_rows = apply_reviews(report["rows"], report["reviews"])
counts = status_counts(merged_rows)
completed, flagged_total = review_progress(report["rows"], report["reviews"])
outstanding = flagged_total - completed

metric_cols = st.columns(6)
metric_cols[0].metric("Findings", f"{len(report['rows']):,}")
metric_cols[1].metric("Approved", f"{counts.get('approved', 0):,}")
metric_cols[2].metric("Rejected", f"{counts.get('rejected', 0):,}")
metric_cols[3].metric("Needs change", f"{counts.get('needs_change', 0):,}")
metric_cols[4].metric(
    "Reused decisions",
    f"{sum(bool(review.get('memory_match')) for review in report['reviews'].values()):,}",
)
metric_cols[5].metric("Review remaining", f"{outstanding:,}")

page = st.radio(
    "Workspace",
    [
        "Review report",
        "Report Library",
        "Decision Memory",
        "Pattern Lab",
        "Rulebook",
    ],
    horizontal=True,
    label_visibility="collapsed",
)


if page == "Review report":
    overview_tab, review_tab, export_tab, detail_tab = st.tabs(
        ["Overview", "Human review", "Exports", "Detailed data"]
    )

    with overview_tab:
        reused_reviews = [
            review
            for review in report["reviews"].values()
            if review.get("memory_match")
        ]
        if reused_reviews:
            exact_reused = sum(
                review.get("memory_match") == "exact" for review in reused_reviews
            )
            near_reused = len(reused_reviews) - exact_reused
            st.info(
                f"Decision Memory prefilled {len(reused_reviews):,} reviews: "
                f"{exact_reused:,} exact contextual matches and "
                f"{near_reused:,} near matches. Every reused decision remains editable."
            )
        if not st.session_state.memory_error:
            similar_reports = report_similarity(
                MEMORY_PATH, report["report_id"], limit=5
            )
            if similar_reports:
                st.markdown("#### Related saved reports")
                similarity_frame = pd.DataFrame(
                    [
                        {
                            "Report": item["report"],
                            "Exact matching findings": item["exact_findings"],
                            "Exact overlap": item["exact_overlap"],
                            "Shared correction pairs": item[
                                "shared_correction_pairs"
                            ],
                        }
                        for item in similar_reports
                    ]
                )
                st.dataframe(
                    similarity_frame,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Exact overlap": st.column_config.ProgressColumn(
                            "Exact overlap",
                            min_value=0.0,
                            max_value=1.0,
                            format="percent",
                        )
                    },
                )
            review_activity = list_report_review_activity(
                MEMORY_PATH, report["report_id"], limit=25
            )
            if review_activity:
                with st.expander("Reviewer activity"):
                    st.caption(
                        "Identity and workflow history stay in the shared audit "
                        "log and are never added to the final training CSV."
                    )
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Reviewer": item["reviewer_name"],
                                    "Email": item["reviewer_email"],
                                    "Action": item["action"].replace("_", " ").title(),
                                    "Issue": item["issue_id"],
                                    "Decision": item["decision"],
                                    "Time": item["occurred_at"],
                                }
                                for item in review_activity
                            ]
                        ),
                        width="stretch",
                        hide_index=True,
                    )
        summary = (
            pd.DataFrame(report["rows"])
            .groupby(["checker", "status"])
            .size()
            .unstack(fill_value=0)
        )
        for status in ("approved", "rejected", "needs_change"):
            if status not in summary:
                summary[status] = 0
        st.dataframe(
            summary[["approved", "rejected", "needs_change"]],
            width="stretch",
        )
        if flagged_total:
            st.markdown("#### Human-review progress")
            st.progress(completed / flagged_total)
            st.caption(f"{completed:,} of {flagged_total:,} flagged decisions saved.")

    with review_tab:
        full_frame = make_review_frame(report)
        if full_frame.empty:
            st.success("This report has no findings requiring human review.")
        else:
            filter_col, search_col = st.columns([1, 2])
            checker_options = sorted(full_frame["checker"].unique().tolist())
            selected_checkers = filter_col.multiselect(
                "Checker",
                checker_options,
                default=checker_options,
            )
            query = search_col.text_input(
                "Search original, proposed, or location",
                placeholder="e.g. Desmos, Unit 4, capitalization",
            )
            visible = full_frame[full_frame["checker"].isin(selected_checkers)].copy()
            if query.strip():
                needle = query.strip().lower()
                haystack = (
                    visible[["original", "proposed", "location"]]
                    .fillna("")
                    .astype(str)
                    .agg(" ".join, axis=1)
                    .str.lower()
                )
                visible = visible[haystack.str.contains(re.escape(needle), regex=True)]

            st.caption(
                f"Showing {len(visible):,} of {len(full_frame):,} flagged findings. "
                "Use a note when choosing needs_change. Write only a concrete "
                "correction or reusable QA signal; workflow comments are omitted "
                "from the final CSV."
            )
            edited = st.data_editor(
                visible,
                width="stretch",
                height=590,
                hide_index=True,
                disabled=[
                    "checker",
                    "field",
                    "original",
                    "proposed",
                    "context",
                    "location",
                    "reused_from",
                    "match_quality",
                    "node_link",
                    "issue_id",
                ],
                column_config={
                    "decision": st.column_config.SelectboxColumn(
                        "Decision",
                        options=DECISION_OPTIONS,
                        required=False,
                        width="medium",
                    ),
                    "review_note": st.column_config.TextColumn(
                        "Correction / QA-training note",
                        width="large",
                        help=(
                            "Examples: “Valid curriculum term,” “Keep lowercase "
                            "because this is the imaginary unit,” or “Use an "
                            "ellipsis.” Confidence and human-review notes are "
                            "removed from the final CSV."
                        ),
                    ),
                    "original": st.column_config.TextColumn("Original", width="medium"),
                    "proposed": st.column_config.TextColumn("Proposed", width="medium"),
                    "context": st.column_config.TextColumn("Context", width="large"),
                    "location": st.column_config.TextColumn("Location", width="large"),
                    "reused_from": st.column_config.TextColumn(
                        "Reused from", width="medium"
                    ),
                    "match_quality": st.column_config.TextColumn(
                        "Memory match", width="small"
                    ),
                    "node_link": st.column_config.LinkColumn(
                        "Open node", display_text="Open"
                    ),
                    "issue_id": None,
                },
                key=f"review-editor-{selected_key}",
            )
            if st.button("Save visible decisions", type="primary"):
                draft_updates = {}
                changed_events = []
                for row in edited.to_dict("records"):
                    issue_id = str(row["issue_id"])
                    decision, note = normalize_ui_decision(
                        str(row.get("decision") or ""),
                        str(row.get("review_note") or ""),
                    )
                    existing_review = report["reviews"].get(issue_id, {})
                    if (
                        decision == existing_review.get("decision", "")
                        and note == existing_review.get("review_note", "")
                    ):
                        report["reviews"][issue_id] = existing_review
                    else:
                        report["reviews"][issue_id] = {
                            "decision": decision,
                            "review_note": note,
                        }
                        changed_events.append(
                            {
                                "issue_id": issue_id,
                                "decision": decision,
                                "detail": {"review_note": note},
                            }
                        )
                    draft_updates[issue_id] = report["reviews"][issue_id]
                if (
                    not st.session_state.memory_error
                    and report.get("library_saved")
                ):
                    draft_result = save_draft_reviews(
                        MEMORY_PATH,
                        report["report_id"],
                        draft_updates,
                    )
                    st.success(
                        f"Saved {draft_result['saved']:,} decisions to the shared "
                        "report draft."
                    )
                else:
                    st.success(f"Saved {len(edited):,} visible rows in this session.")
                record_report_activity(
                    report,
                    "save_decisions",
                    changed_events,
                    current_user,
                )
                st.rerun()

            with st.expander("Import a reviewed CSV instead"):
                imported_review = st.file_uploader(
                    "Reviewed flagged_for_review CSV",
                    type=["csv"],
                    key=f"review-import-{selected_key}",
                )
                if st.button(
                    "Import decisions",
                    disabled=imported_review is None,
                    key=f"review-import-button-{selected_key}",
                ):
                    try:
                        before_import = {
                            issue_id: dict(review)
                            for issue_id, review in report["reviews"].items()
                        }
                        applied, unmatched = import_review_sheet(
                            imported_review.getvalue(), report
                        )
                        if (
                            not st.session_state.memory_error
                            and report.get("library_saved")
                        ):
                            save_draft_reviews(
                                MEMORY_PATH,
                                report["report_id"],
                                report["reviews"],
                            )
                        imported_events = [
                            {
                                "issue_id": issue_id,
                                "decision": str(review.get("decision") or ""),
                                "detail": {
                                    "review_note": str(
                                        review.get("review_note") or ""
                                    )
                                },
                            }
                            for issue_id, review in report["reviews"].items()
                            if review != before_import.get(issue_id, {})
                        ]
                        record_report_activity(
                            report,
                            "import_review_sheet",
                            imported_events,
                            current_user,
                        )
                        if unmatched:
                            st.warning(
                                f"Imported {applied:,} decisions; ignored "
                                f"{unmatched:,} unmatched IDs."
                            )
                        else:
                            st.success(f"Imported {applied:,} decisions.")
                        st.rerun()
                    except QAEngineError as error:
                        st.error(str(error))

    with export_tab:
        merged_rows = apply_reviews(report["rows"], report["reviews"])
        completed, flagged_total = review_progress(report["rows"], report["reviews"])
        outstanding = flagged_total - completed
        if outstanding:
            st.warning(
                f"{outstanding:,} human decisions are still blank. The current "
                "final CSV keeps those rows as needs_change placeholders."
            )
        else:
            human_changes = sum(
                1
                for issue_id, review in report["reviews"].items()
                if review.get("decision") == "needs_change"
            )
            st.success(
                "Human review is complete. "
                f"{human_changes:,} reviewer-confirmed changes remain as needs_change."
            )

        training_rows = finalize_training_comments(merged_rows)
        raw_comment_count = sum(
            bool(str(row.get("comment") or "").strip()) for row in merged_rows
        )
        kept_comment_rows = [
            row for row in training_rows if str(row.get("comment") or "").strip()
        ]
        suppressed_comments = raw_comment_count - len(kept_comment_rows)
        st.caption(
            "Final-comment policy: "
            f"{len(kept_comment_rows):,} concrete QA-training comments will be "
            f"included; {suppressed_comments:,} confidence, workflow, ambiguity, "
            "or human-review comments will be blanked. Detailed data keeps the "
            "internal reviewer guidance."
        )
        if kept_comment_rows:
            with st.expander("Preview comments included in the final CSV"):
                st.dataframe(
                    pd.DataFrame(kept_comment_rows)[
                        [
                            "status",
                            "checker",
                            "original",
                            "proposed",
                            "comment",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

        publish_col, publish_note_col = st.columns([1, 2])
        if publish_col.button(
            "Publish review to shared memory",
            type="primary",
            disabled=bool(outstanding or st.session_state.memory_error),
            width="stretch",
            help=(
                "Makes this report's completed human decisions reusable by "
                "matching reports on this app deployment."
            ),
        ):
            if report.get("library_saved"):
                save_draft_reviews(
                    MEMORY_PATH,
                    report["report_id"],
                    report["reviews"],
                )
            result = publish_report_reviews(
                MEMORY_PATH,
                report["report_id"],
                report["filename"],
                report["rows"],
                report["reviews"],
            )
            report["memory_published"] = result["published"]
            record_report_activity(
                report,
                "publish_to_memory",
                [
                    {
                        "detail": {
                            "published": result["published"],
                            "skipped_blank": result["skipped_blank"],
                            "skipped_unmatchable": result[
                                "skipped_unmatchable"
                            ],
                        }
                    }
                ],
                current_user,
            )
            newly_seeded = seed_all_loaded_reports()
            st.session_state.memory_publish_flash = {
                **result,
                "newly_seeded": newly_seeded["reused"],
            }
            st.rerun()
        if outstanding:
            publish_note_col.caption(
                "Complete every flagged decision before publishing to shared memory."
            )
        elif report.get("memory_published") is not None:
            publish_note_col.caption(
                f"{report['memory_published']:,} reviewed findings from this "
                "report are in shared memory."
            )
        else:
            publish_note_col.caption(
                "Publishing stores compact contextual fingerprints and decisions, "
                "not the original HTML file."
            )

        publish_flash = st.session_state.pop("memory_publish_flash", None)
        if publish_flash:
            message = (
                f"Published {publish_flash['published']:,} decisions to shared memory."
            )
            if publish_flash["newly_seeded"]:
                message += (
                    f" Prefilled {publish_flash['newly_seeded']:,} blank reviews "
                    "in other loaded reports."
                )
            st.success(message)

        st.divider()
        download_cols = st.columns(2)
        download_cols[0].download_button(
            "Download final Kiddom CSV",
            final_csv_bytes(merged_rows),
            file_name=f"{report['name']}_FINAL.csv",
            mime="text/csv",
            width="stretch",
        )
        download_cols[1].download_button(
            "Download review sheet",
            review_csv_bytes(report["rows"], report["reviews"]),
            file_name=f"{report['name']}_flagged_for_review.csv",
            mime="text/csv",
            width="stretch",
        )
        download_cols[0].download_button(
            "Download detailed audit CSV",
            detailed_csv_bytes(report["rows"]),
            file_name=f"{report['name']}_detailed.csv",
            mime="text/csv",
            width="stretch",
        )
        download_cols[1].download_button(
            "Download complete review package",
            build_export_zip(report),
            file_name=f"{report['name']}_review_package.zip",
            mime="application/zip",
            width="stretch",
        )
        render_jira_handoff(
            report,
            merged_rows,
            outstanding,
            jira_config,
            jira_config_error,
            current_user,
        )

    with detail_tab:
        detail_frame = pd.DataFrame(report["rows"])
        st.dataframe(
            detail_frame,
            width="stretch",
            height=640,
            hide_index=True,
            column_config={
                "node_link": st.column_config.LinkColumn(
                    "Node", display_text="Open"
                )
            },
        )


elif page == "Report Library":
    st.markdown(
        """
        <div class="qa-callout">
          Every uploaded HTML becomes a compact, persistent report snapshot.
          Reviewers can reopen it later, resume shared draft decisions, and see
          which other state or course editions contain the same findings. The
          original HTML binary is not retained.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.memory_error:
        st.error(f"Report Library unavailable: {st.session_state.memory_error}")
    else:
        library_summary = library_stats(MEMORY_PATH)
        library_cols = st.columns(4)
        library_cols[0].metric("Saved reports", f"{library_summary['reports']:,}")
        library_cols[1].metric(
            "Parsed findings", f"{library_summary['findings']:,}"
        )
        library_cols[2].metric(
            "Recurring findings", f"{library_summary['recurring_findings']:,}"
        )
        library_cols[3].metric(
            "Saved draft decisions", f"{library_summary['draft_reviews']:,}"
        )

        library_reports = list_report_library(MEMORY_PATH)
        if library_reports:
            st.markdown("#### Saved reports")
            library_frame = pd.DataFrame(
                [
                    {
                        "Report": item["report_name"],
                        "Findings": item["finding_count"],
                        "Flagged at upload": item["reviewable_count"],
                        "Saved decisions": item["draft_reviews"],
                        "Published decisions": item["published_reviews"],
                        "Uploads": item["upload_count"],
                        "Last uploaded": item["last_uploaded_at"],
                    }
                    for item in library_reports
                ]
            )
            st.dataframe(library_frame, width="stretch", hide_index=True)

            library_lookup = {
                str(item["report_id"]): item for item in library_reports
            }
            open_col, button_col = st.columns([3, 1])
            report_to_open = open_col.selectbox(
                "Saved report",
                options=list(library_lookup),
                format_func=lambda report_id: library_lookup[report_id][
                    "report_name"
                ],
                key="library-page-report-select",
            )
            if button_col.button(
                "Open report",
                type="primary",
                width="stretch",
                key="library-page-open",
            ):
                open_saved_report(str(report_to_open))
                st.rerun()
        else:
            st.info("Upload the first HTML report to start the shared library.")

        related = report_similarity(MEMORY_PATH, report["report_id"])
        st.markdown("#### Related to the active report")
        if related:
            related_frame = pd.DataFrame(
                [
                    {
                        "Report": item["report"],
                        "Exact matching findings": item["exact_findings"],
                        "Exact overlap": item["exact_overlap"],
                        "Shared correction pairs": item[
                            "shared_correction_pairs"
                        ],
                        "Correction-pair overlap": item[
                            "correction_pair_overlap"
                        ],
                    }
                    for item in related
                ]
            )
            st.dataframe(
                related_frame,
                width="stretch",
                hide_index=True,
                column_config={
                    "Exact overlap": st.column_config.ProgressColumn(
                        min_value=0.0,
                        max_value=1.0,
                        format="percent",
                    ),
                    "Correction-pair overlap": st.column_config.ProgressColumn(
                        min_value=0.0,
                        max_value=1.0,
                        format="percent",
                    ),
                },
            )
        else:
            st.caption(
                "No related saved report has been found yet. Similarity will "
                "appear as more courses are uploaded."
            )


elif page == "Decision Memory":
    st.markdown(
        """
        <div class="qa-callout">
          Decision Memory carries completed reviews across nearly identical
          courses—for example, from IM v.360 Grade 2 into New Mexico Grade 2.
          Matching uses checker, field, original/proposed text, and normalized
          surrounding context—not issue IDs, state/course prefixes, or course
          URLs. Conflicting prior reviews are never reused.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.memory_error:
        st.error(f"Decision memory unavailable: {st.session_state.memory_error}")
    else:
        memory_summary = memory_stats(MEMORY_PATH)
        memory_cols = st.columns(4)
        memory_cols[0].metric(
            "Published observations", f"{memory_summary['observations']:,}"
        )
        memory_cols[1].metric(
            "Reusable findings", f"{memory_summary['findings']:,}"
        )
        memory_cols[2].metric(
            "Source reports", f"{memory_summary['reports']:,}"
        )
        memory_cols[3].metric(
            "Conflicting findings", f"{memory_summary['conflicts']:,}"
        )

        action_col, download_col = st.columns(2)
        if action_col.button(
            "Apply memory to blank reviews",
            width="stretch",
        ):
            result = seed_all_loaded_reports()
            st.success(
                f"Reused {result['reused']:,} decisions "
                f"({result['exact']:,} exact, {result['near']:,} near)."
            )
            st.rerun()
        download_col.download_button(
            "Download portable shared memory",
            export_memory_bytes(MEMORY_PATH),
            file_name="kiddom_qa_shared_memory.json",
            mime="application/json",
            width="stretch",
        )

        with st.expander("Merge decision memory from another deployment"):
            memory_upload = st.file_uploader(
                "Decision-memory JSON",
                type=["json"],
                key="decision-memory-import",
            )
            if st.button(
                "Merge uploaded memory",
                disabled=memory_upload is None,
                key="decision-memory-import-button",
            ):
                try:
                    result = import_memory_bytes(
                        MEMORY_PATH, memory_upload.getvalue()
                    )
                    seeded = seed_all_loaded_reports()
                    st.success(
                        f"Imported {result['reports']:,} reports, "
                        f"{result['findings']:,} parsed findings, and "
                        f"{result['imported']:,} decisions; prefilled "
                        f"{seeded['reused']:,} blank reviews."
                    )
                    st.rerun()
                except DecisionMemoryError as error:
                    st.error(str(error))

        st.markdown(
            """
            **How reuse works**

            - **Exact match:** the normalized surrounding content is identical.
            - **Near match:** the checker and correction pair are identical and
              surrounding content is at least 97% similar.
            - **Conflict:** prior reviewers disagreed, so the app leaves the new
              decision blank.

            The SQLite database is shared by users of the same deployed app.
            Set `KIDDOM_SHARED_MEMORY_PATH` to a mounted persistent volume so
            report snapshots, drafts, and decisions survive redeployments.
            """
        )


elif page == "Pattern Lab":
    st.markdown(
        """
        <div class="qa-callout">
          Published human decisions provide the label; the shared Report
          Library shows how often and across how many courses that pattern
          occurs. Unreviewed reports can strengthen coverage evidence but can
          never decide a status. Promotion remains explicit.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.memory_error:
        st.error(f"Pattern evidence unavailable: {st.session_state.memory_error}")
        suggestions = []
    else:
        suggestions = build_shared_pattern_suggestions(
            shared_pattern_evidence(MEMORY_PATH),
            st.session_state.rules,
        )
    if not suggestions:
        st.info(
            "Publish at least one completed human review. As more reports are "
            "uploaded, this screen will show the reach of each confirmed pattern."
        )
    else:
        evidence_col, coverage_col = st.columns(2)
        max_evidence = max(item["evidence_count"] for item in suggestions)
        max_coverage = max(item["course_coverage"] for item in suggestions)
        if max_evidence > 1:
            min_evidence = evidence_col.slider(
                "Minimum supporting decisions",
                min_value=1,
                max_value=max_evidence,
                value=1,
            )
        else:
            evidence_col.metric("Human decision range", "1")
            min_evidence = 1
        if max_coverage > 1:
            min_coverage = coverage_col.slider(
                "Minimum course coverage",
                min_value=1,
                max_value=max_coverage,
                value=min(2, max_coverage),
            )
        else:
            coverage_col.metric("Course coverage range", "1")
            min_coverage = 1
        shown = [
            item
            for item in suggestions
            if item["evidence_count"] >= min_evidence
            and item["course_coverage"] >= min_coverage
        ]
        if not shown:
            st.info("No pattern meets both evidence filters.")
            chosen = []
        else:
            suggestion_frame = pd.DataFrame(shown)
            edited_suggestions = st.data_editor(
                suggestion_frame,
                width="stretch",
                height=540,
                hide_index=True,
                disabled=[
                    column
                    for column in suggestion_frame.columns
                    if column != "selected"
                ],
                column_config={
                    "selected": st.column_config.CheckboxColumn(
                        "Promote", default=False
                    ),
                    "suggestion_id": None,
                    "evidence_count": st.column_config.NumberColumn(
                        "Human decisions", format="%d"
                    ),
                    "human_reports": st.column_config.NumberColumn(
                        "Reviewed courses", format="%d"
                    ),
                    "course_coverage": st.column_config.NumberColumn(
                        "Course coverage", format="%d"
                    ),
                    "occurrences": st.column_config.NumberColumn(
                        "Occurrences", format="%d"
                    ),
                },
                key="pattern-suggestion-editor",
            )
            chosen = edited_suggestions[
                edited_suggestions["selected"] == True  # noqa: E712
            ].to_dict("records")
        if st.button(
            f"Promote {len(chosen):,} selected patterns",
            type="primary",
            disabled=not chosen,
        ):
            st.session_state.rules = promote_suggestions(
                st.session_state.rules, chosen
            )
            reclassify_loaded_reports()
            st.success(
                "Patterns promoted and loaded reports reclassified. Export the "
                "rulebook to persist this learning."
            )
            st.rerun()

    st.download_button(
        "Download current rulebook",
        rules_json_bytes(st.session_state.rules),
        file_name="kiddom_qa_rules.json",
        mime="application/json",
    )


else:
    rule_cols = st.columns(3)
    rule_cols[0].metric(
        "Protected spelling terms",
        len(st.session_state.rules["protected_spelling_terms"]),
    )
    rule_cols[1].metric(
        "Exact human-approved rules", len(st.session_state.rules["exact_rules"])
    )
    rule_cols[2].metric(
        "Safe typo targets", len(st.session_state.rules["safe_typo_targets"])
    )
    st.markdown(
        """
        <p class="qa-rule-note">
          Base rules are version-controlled. Pattern Lab promotions affect this
          browser session until you download the JSON. Commit that file as the
          app's <code>rules/base_rules.json</code>, or run the included sync
          script to make it available to the Codex skill.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download rulebook JSON",
        rules_json_bytes(st.session_state.rules),
        file_name="kiddom_qa_rules.json",
        mime="application/json",
        type="primary",
    )
    with st.expander("Protected terms"):
        st.write(", ".join(st.session_state.rules["protected_spelling_terms"]))
    with st.expander("Exact rules"):
        exact_rules = st.session_state.rules["exact_rules"]
        if exact_rules:
            st.dataframe(pd.DataFrame(exact_rules), width="stretch")
        else:
            st.caption("No exact feedback rules have been promoted yet.")
