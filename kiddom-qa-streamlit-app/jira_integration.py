from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import quote, urlparse

import requests


ISSUE_FIELDS = [
    "summary",
    "status",
    "assignee",
    "attachment",
    "description",
    "updated",
]
HTTP_LINK_RE = re.compile(r"https?://[^\s<>\"]+")


class JiraIntegrationError(RuntimeError):
    """A safe, user-facing Jira integration error."""


class JiraHandoffError(JiraIntegrationError):
    """A handoff failure that records any steps Jira already accepted."""

    def __init__(self, message: str, completed_steps: list[dict[str, str]]):
        super().__init__(message)
        self.completed_steps = completed_steps


@dataclass(frozen=True)
class JiraConfig:
    base_url: str
    user_email: str
    api_token: str
    project_key: str = ""
    ready_for_qa_status: str = "Ready for QA"
    qa_account_id: str = ""
    ticket_jql: str = ""
    max_results: int = 50

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "JiraConfig":
        base_url = str(values.get("JIRA_BASE_URL") or "").strip().rstrip("/")
        user_email = str(values.get("JIRA_USER_EMAIL") or "").strip()
        api_token = str(values.get("JIRA_API_TOKEN") or "").strip()
        missing = [
            name
            for name, value in (
                ("JIRA_BASE_URL", base_url),
                ("JIRA_USER_EMAIL", user_email),
                ("JIRA_API_TOKEN", api_token),
            )
            if not value
        ]
        if missing:
            raise JiraIntegrationError(
                "Jira is not configured. Missing " + ", ".join(missing) + "."
            )
        parsed = urlparse(base_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise JiraIntegrationError(
                "JIRA_BASE_URL must be a complete HTTPS Jira Cloud URL."
            )
        if parsed.query or parsed.fragment:
            raise JiraIntegrationError(
                "JIRA_BASE_URL cannot include a query string or fragment."
            )
        try:
            max_results = int(values.get("JIRA_MAX_RESULTS") or 50)
        except (TypeError, ValueError) as error:
            raise JiraIntegrationError(
                "JIRA_MAX_RESULTS must be a whole number."
            ) from error
        if not 1 <= max_results <= 100:
            raise JiraIntegrationError(
                "JIRA_MAX_RESULTS must be between 1 and 100."
            )
        return cls(
            base_url=base_url,
            user_email=user_email,
            api_token=api_token,
            project_key=str(values.get("JIRA_PROJECT_KEY") or "").strip(),
            ready_for_qa_status=(
                str(values.get("JIRA_READY_FOR_QA_STATUS") or "Ready for QA").strip()
            ),
            qa_account_id=str(values.get("JIRA_QA_ACCOUNT_ID") or "").strip(),
            ticket_jql=str(values.get("JIRA_TICKET_JQL") or "").strip(),
            max_results=max_results,
        )


def _jira_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _adf_text_and_links(value: Any) -> tuple[str, list[str]]:
    text_parts: list[str] = []
    links: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, str):
            text_parts.append(node)
            links.extend(HTTP_LINK_RE.findall(node))
            return
        if isinstance(node, list):
            for child in node:
                visit(child)
            return
        if not isinstance(node, Mapping):
            return

        text = node.get("text")
        if isinstance(text, str):
            text_parts.append(text)
            links.extend(HTTP_LINK_RE.findall(text))

        attrs = node.get("attrs")
        if isinstance(attrs, Mapping):
            for key in ("href", "url"):
                link = attrs.get(key)
                if isinstance(link, str) and link.startswith(("http://", "https://")):
                    links.append(link)

        marks = node.get("marks")
        if isinstance(marks, list):
            for mark in marks:
                if not isinstance(mark, Mapping):
                    continue
                mark_attrs = mark.get("attrs")
                if not isinstance(mark_attrs, Mapping):
                    continue
                link = mark_attrs.get("href")
                if isinstance(link, str) and link.startswith(("http://", "https://")):
                    links.append(link)

        content = node.get("content")
        if isinstance(content, list):
            for child in content:
                visit(child)

    visit(value)
    unique_links = list(dict.fromkeys(link.rstrip(".,);]") for link in links))
    return " ".join(part.strip() for part in text_parts if part.strip()), unique_links


def _normalized_attachment(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(raw.get("id") or ""),
        "filename": str(raw.get("filename") or "attachment"),
        "mime_type": str(raw.get("mimeType") or ""),
        "size": int(raw.get("size") or 0),
        "content_url": str(raw.get("content") or ""),
    }


def normalize_issue(raw: Mapping[str, Any], base_url: str) -> dict[str, Any]:
    fields = raw.get("fields")
    fields = fields if isinstance(fields, Mapping) else {}
    status = fields.get("status")
    status = status if isinstance(status, Mapping) else {}
    assignee = fields.get("assignee")
    assignee = assignee if isinstance(assignee, Mapping) else {}
    description_text, description_links = _adf_text_and_links(
        fields.get("description")
    )
    attachments = fields.get("attachment")
    attachments = attachments if isinstance(attachments, list) else []
    key = str(raw.get("key") or "")
    return {
        "id": str(raw.get("id") or ""),
        "key": key,
        "summary": str(fields.get("summary") or ""),
        "status": str(status.get("name") or ""),
        "assignee_name": str(assignee.get("displayName") or ""),
        "assignee_account_id": str(assignee.get("accountId") or ""),
        "updated": str(fields.get("updated") or ""),
        "description": description_text,
        "links": description_links,
        "attachments": [
            _normalized_attachment(item)
            for item in attachments
            if isinstance(item, Mapping)
        ],
        "browse_url": f"{base_url}/browse/{quote(key, safe='-')}",
    }


def is_html_attachment(attachment: Mapping[str, Any]) -> bool:
    filename = str(attachment.get("filename") or "").lower()
    mime_type = str(attachment.get("mime_type") or "").lower()
    return filename.endswith((".html", ".htm")) or mime_type in {
        "text/html",
        "application/xhtml+xml",
    }


class JiraClient:
    def __init__(
        self,
        config: JiraConfig,
        session: requests.Session | None = None,
    ):
        self.config = config
        self.session = session or requests.Session()

    def _request(
        self,
        method: str,
        path: str,
        *,
        expected: tuple[int, ...] = (200,),
        **kwargs: Any,
    ) -> requests.Response:
        headers = {"Accept": "application/json", **kwargs.pop("headers", {})}
        try:
            response = self.session.request(
                method,
                f"{self.config.base_url}{path}",
                auth=(self.config.user_email, self.config.api_token),
                headers=headers,
                timeout=(10, 90),
                **kwargs,
            )
        except requests.RequestException as error:
            raise JiraIntegrationError(
                f"Jira could not be reached: {error.__class__.__name__}."
            ) from error
        if response.status_code not in expected:
            detail = ""
            try:
                body = response.json()
                if isinstance(body, Mapping):
                    messages = body.get("errorMessages") or []
                    errors = body.get("errors") or {}
                    parts = [str(item) for item in messages if item]
                    if isinstance(errors, Mapping):
                        parts.extend(
                            f"{field}: {message}"
                            for field, message in errors.items()
                            if message
                        )
                    detail = "; ".join(parts)
            except (ValueError, TypeError):
                detail = ""
            suffix = f" {detail}" if detail else ""
            raise JiraIntegrationError(
                f"Jira returned HTTP {response.status_code}.{suffix}"
            )
        return response

    def test_connection(self) -> dict[str, str]:
        response = self._request("GET", "/rest/api/3/myself")
        data = response.json()
        return {
            "account_id": str(data.get("accountId") or ""),
            "display_name": str(data.get("displayName") or ""),
        }

    def search_users(self, query: str, max_results: int = 50) -> list[dict[str, str]]:
        response = self._request(
            "GET",
            "/rest/api/3/user/search",
            params={
                "query": query.strip(),
                "maxResults": min(max(int(max_results), 1), 100),
            },
        )
        data = response.json()
        if not isinstance(data, list):
            raise JiraIntegrationError("Jira returned an unexpected user-search result.")
        return [
            {
                "account_id": str(item.get("accountId") or ""),
                "display_name": str(item.get("displayName") or "Unnamed Jira user"),
                "email": str(item.get("emailAddress") or ""),
                "active": bool(item.get("active", True)),
            }
            for item in data
            if isinstance(item, Mapping) and item.get("accountId")
        ]

    def find_user_for_identity(self, email: str, name: str = "") -> dict[str, str]:
        normalized_email = email.strip().casefold()
        normalized_name = name.strip().casefold()
        candidates: dict[str, dict[str, str]] = {}
        for query in dict.fromkeys(
            value for value in (email.strip(), name.strip()) if value
        ):
            for person in self.search_users(query):
                if person["active"]:
                    candidates[person["account_id"]] = person
            exact_email = next(
                (
                    person
                    for person in candidates.values()
                    if person["email"].strip().casefold() == normalized_email
                ),
                None,
            )
            if exact_email:
                return exact_email

        if len(candidates) == 1:
            return next(iter(candidates.values()))
        exact_name_matches = [
            person
            for person in candidates.values()
            if normalized_name
            and person["display_name"].strip().casefold() == normalized_name
        ]
        if len(exact_name_matches) == 1:
            return exact_name_matches[0]
        if not candidates:
            raise JiraIntegrationError(
                "No active Jira user matches your Google Workspace account. "
                "Ask the app administrator to add your email to [jira_user_map]."
            )
        raise JiraIntegrationError(
            "More than one Jira user could match your Google Workspace account. "
            "Ask the app administrator to add your email to [jira_user_map]."
        )

    def _assigned_jql(self, account_id: str) -> str:
        if self.config.ticket_jql:
            return (
                self.config.ticket_jql.replace(
                    "{account_id}", account_id.replace('"', '\\"')
                ).replace("{project_key}", self.config.project_key)
            )
        clauses = [
            f"assignee = {_jira_quote(account_id)}",
            "statusCategory != Done",
        ]
        if self.config.project_key:
            clauses.insert(0, f"project = {_jira_quote(self.config.project_key)}")
        return " AND ".join(clauses) + " ORDER BY updated DESC"

    def search_assigned_issues(self, account_id: str) -> list[dict[str, Any]]:
        response = self._request(
            "POST",
            "/rest/api/3/search/jql",
            headers={"Content-Type": "application/json"},
            json={
                "jql": self._assigned_jql(account_id),
                "maxResults": self.config.max_results,
                "fields": ISSUE_FIELDS,
            },
        )
        data = response.json()
        issues = data.get("issues") if isinstance(data, Mapping) else None
        if not isinstance(issues, list):
            raise JiraIntegrationError("Jira returned an unexpected ticket search result.")
        return [
            normalize_issue(item, self.config.base_url)
            for item in issues
            if isinstance(item, Mapping)
        ]

    def get_issue(self, issue_key: str) -> dict[str, Any]:
        response = self._request(
            "GET",
            f"/rest/api/3/issue/{quote(issue_key, safe='-')}",
            params={"fields": ",".join(ISSUE_FIELDS)},
        )
        data = response.json()
        if not isinstance(data, Mapping):
            raise JiraIntegrationError("Jira returned an unexpected ticket result.")
        return normalize_issue(data, self.config.base_url)

    def download_attachment(
        self, attachment_id: str, max_bytes: int = 1_200_000_000
    ) -> bytes:
        response = self._request(
            "GET",
            f"/rest/api/3/attachment/content/{quote(str(attachment_id), safe='')}",
            params={"redirect": "false"},
            expected=(200, 206),
            stream=True,
        )
        declared_size = int(response.headers.get("Content-Length") or 0)
        if declared_size > max_bytes:
            raise JiraIntegrationError(
                "The Jira HTML attachment is larger than this app's 1.2 GB limit."
            )
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise JiraIntegrationError(
                    "The Jira HTML attachment is larger than this app's 1.2 GB limit."
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def add_attachment(
        self,
        issue_key: str,
        filename: str,
        payload: bytes,
        mime_type: str = "text/csv",
    ) -> list[dict[str, Any]]:
        response = self._request(
            "POST",
            f"/rest/api/3/issue/{quote(issue_key, safe='-')}/attachments",
            expected=(200,),
            headers={"X-Atlassian-Token": "no-check"},
            files={"file": (filename, payload, mime_type)},
        )
        data = response.json()
        return data if isinstance(data, list) else []

    def get_transitions(self, issue_key: str) -> list[dict[str, Any]]:
        response = self._request(
            "GET",
            f"/rest/api/3/issue/{quote(issue_key, safe='-')}/transitions",
        )
        data = response.json()
        transitions = data.get("transitions") if isinstance(data, Mapping) else None
        if not isinstance(transitions, list):
            raise JiraIntegrationError("Jira returned an unexpected transition list.")
        return [item for item in transitions if isinstance(item, Mapping)]

    def transition_issue(self, issue_key: str, target_status: str) -> str:
        transitions = self.get_transitions(issue_key)
        target = target_status.strip().casefold()
        match = next(
            (
                transition
                for transition in transitions
                if str((transition.get("to") or {}).get("name") or "")
                .strip()
                .casefold()
                == target
            ),
            None,
        )
        if match is None:
            available = ", ".join(
                sorted(
                    {
                        (
                            f"{str(item.get('name') or '').strip()} → "
                            f"{str((item.get('to') or {}).get('name') or '').strip()}"
                        ).strip(" →")
                        for item in transitions
                        if item.get("name")
                    }
                )
            )
            hint = f" Available transitions: {available}." if available else ""
            raise JiraIntegrationError(
                f'No Jira transition leads to "{target_status}".{hint}'
            )
        transition_id = str(match.get("id") or "")
        self._request(
            "POST",
            f"/rest/api/3/issue/{quote(issue_key, safe='-')}/transitions",
            expected=(204,),
            headers={"Content-Type": "application/json"},
            json={"transition": {"id": transition_id}},
        )
        return str((match.get("to") or {}).get("name") or match.get("name") or "")

    def assign_issue(self, issue_key: str, account_id: str) -> None:
        self._request(
            "PUT",
            f"/rest/api/3/issue/{quote(issue_key, safe='-')}/assignee",
            expected=(204,),
            headers={"Content-Type": "application/json"},
            json={"accountId": account_id},
        )

    def handoff_completed_review(
        self,
        issue_key: str,
        filename: str,
        csv_payload: bytes,
        qa_account_id: str,
        target_status: str,
    ) -> list[dict[str, str]]:
        """Attach, transition, and reassign with retry-safe checks."""
        completed: list[dict[str, str]] = []
        try:
            issue = self.get_issue(issue_key)
            existing_names = {
                str(item.get("filename") or "")
                for item in issue.get("attachments", [])
            }
            if filename in existing_names:
                completed.append(
                    {"step": "CSV attachment", "result": "Already attached"}
                )
            else:
                self.add_attachment(issue_key, filename, csv_payload)
                completed.append({"step": "CSV attachment", "result": "Attached"})

            if str(issue.get("status") or "").casefold() == target_status.casefold():
                completed.append(
                    {"step": "Jira status", "result": f"Already {target_status}"}
                )
            else:
                new_status = self.transition_issue(issue_key, target_status)
                completed.append(
                    {"step": "Jira status", "result": f"Moved to {new_status}"}
                )

            if str(issue.get("assignee_account_id") or "") == qa_account_id:
                completed.append(
                    {"step": "QA assignee", "result": "Already assigned"}
                )
            else:
                self.assign_issue(issue_key, qa_account_id)
                completed.append({"step": "QA assignee", "result": "Reassigned"})
        except JiraIntegrationError as error:
            raise JiraHandoffError(str(error), completed) from error
        return completed

    def browse_url(self, issue_key: str) -> str:
        return f"{self.config.base_url}/browse/{quote(issue_key, safe='-')}"
