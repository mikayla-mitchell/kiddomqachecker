from __future__ import annotations

import pytest

from jira_integration import (
    JiraClient,
    JiraConfig,
    JiraHandoffError,
    JiraIntegrationError,
    is_html_attachment,
    is_qa_report_issue,
    normalize_issue,
)


class FakeResponse:
    def __init__(
        self,
        status_code=200,
        data=None,
        content=b"",
        headers=None,
    ):
        self.status_code = status_code
        self._data = data
        self._content = content
        self.headers = headers or {}

    def json(self):
        if self._data is None:
            raise ValueError("No JSON")
        return self._data

    def iter_content(self, chunk_size):
        del chunk_size
        yield self._content


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def config(**overrides):
    values = {
        "JIRA_BASE_URL": "https://example.atlassian.net",
        "JIRA_USER_EMAIL": "service@example.com",
        "JIRA_API_TOKEN": "secret",
        "JIRA_PROJECT_KEY": "CURR",
        "JIRA_READY_FOR_QA_STATUS": "Ready for QA",
        "JIRA_QA_ACCOUNT_ID": "qa-123",
    }
    values.update(overrides)
    return JiraConfig.from_mapping(values)


def raw_issue(status="In Review", assignee="reviewer-1", attachments=None):
    return {
        "id": "10001",
        "key": "CURR-42",
        "fields": {
            "summary": "Review IM v.360 Grade 2",
            "status": {"name": status},
            "assignee": {
                "displayName": "Mikayla",
                "accountId": assignee,
            },
            "attachment": attachments or [],
            "updated": "2026-07-29T12:00:00.000+0000",
            "description": {
                "type": "doc",
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {
                                "type": "text",
                                "text": "QA report",
                                "marks": [
                                    {
                                        "type": "link",
                                        "attrs": {
                                            "href": "https://reports.example/report"
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
        },
    }


def test_config_requires_server_side_credentials():
    with pytest.raises(JiraIntegrationError, match="JIRA_API_TOKEN"):
        JiraConfig.from_mapping(
            {
                "JIRA_BASE_URL": "https://example.atlassian.net",
                "JIRA_USER_EMAIL": "service@example.com",
            }
        )
    with pytest.raises(JiraIntegrationError, match="HTTPS"):
        config(JIRA_BASE_URL="http://example.atlassian.net")


def test_user_and_assigned_ticket_search_use_current_cloud_endpoints():
    session = FakeSession(
        [
            FakeResponse(
                data=[
                    {
                        "accountId": "reviewer-1",
                        "displayName": "Mikayla",
                        "emailAddress": "mikayla@example.com",
                        "active": True,
                    }
                ]
            ),
            FakeResponse(data={"isLast": True, "issues": [raw_issue()]}),
        ]
    )
    client = JiraClient(config(), session=session)
    users = client.search_users("Mikayla")
    issues = client.search_assigned_issues(users[0]["account_id"])

    assert users[0]["display_name"] == "Mikayla"
    assert issues[0]["key"] == "CURR-42"
    assert issues[0]["links"] == ["https://reports.example/report"]
    search_call = session.calls[1]
    assert search_call[0:2] == (
        "POST",
        "https://example.atlassian.net/rest/api/3/search/jql",
    )
    assert 'project = "CURR"' in search_call[2]["json"]["jql"]
    assert 'assignee = "reviewer-1"' in search_call[2]["json"]["jql"]


def test_google_identity_resolves_to_exact_jira_user():
    session = FakeSession(
        [
            FakeResponse(
                data=[
                    {
                        "accountId": "reviewer-1",
                        "displayName": "Mikayla Mitchell",
                        "emailAddress": "mikayla@kiddom.co",
                        "active": True,
                    },
                    {
                        "accountId": "reviewer-2",
                        "displayName": "Mikayla M.",
                        "emailAddress": "other@kiddom.co",
                        "active": True,
                    },
                ]
            )
        ]
    )
    reviewer = JiraClient(config(), session=session).find_user_for_identity(
        "mikayla@kiddom.co", "Mikayla Mitchell"
    )
    assert reviewer["account_id"] == "reviewer-1"


def test_short_reviewer_name_prefers_active_first_name_matches():
    session = FakeSession(
        [
            FakeResponse(
                data=[
                    {
                        "accountId": "steve-1",
                        "displayName": "Steve Reviewer",
                        "active": True,
                    },
                    {
                        "accountId": "stephen-1",
                        "displayName": "Stephen Reviewer",
                        "active": True,
                    },
                    {
                        "accountId": "steve-old",
                        "displayName": "Steve Former",
                        "active": False,
                    },
                ]
            )
        ]
    )
    people = JiraClient(config(), session=session).search_named_users("Steve")

    assert [person["account_id"] for person in people] == ["steve-1"]


def test_ambiguous_google_identity_requires_admin_mapping():
    session = FakeSession(
        [
            FakeResponse(
                data=[
                    {
                        "accountId": "one",
                        "displayName": "Same Person",
                        "active": True,
                    },
                    {
                        "accountId": "two",
                        "displayName": "Same Person",
                        "active": True,
                    },
                ]
            ),
            FakeResponse(data=[]),
        ]
    )
    with pytest.raises(JiraIntegrationError, match="jira_user_map"):
        JiraClient(config(), session=session).find_user_for_identity(
            "hidden@kiddom.co", "Same Person"
        )


def test_html_attachment_detection_and_authenticated_download():
    attachment = {
        "id": "9001",
        "filename": "course_review_detailed.html",
        "mime_type": "application/octet-stream",
    }
    assert is_html_attachment(attachment)
    session = FakeSession(
        [FakeResponse(content=b"<html>report</html>", headers={"Content-Length": "19"})]
    )
    client = JiraClient(config(), session=session)
    assert client.download_attachment("9001") == b"<html>report</html>"
    call = session.calls[0]
    assert call[1].endswith("/rest/api/3/attachment/content/9001")
    assert call[2]["auth"] == ("service@example.com", "secret")


def test_normalize_issue_extracts_adf_links_and_attachment_metadata():
    issue = normalize_issue(
        raw_issue(
            attachments=[
                {
                    "id": 9001,
                    "filename": "report.html",
                    "mimeType": "text/html",
                    "size": 123,
                }
            ]
        ),
        "https://example.atlassian.net",
    )
    assert issue["description"] == "QA report"
    assert issue["links"] == ["https://reports.example/report"]
    assert issue["attachments"][0]["filename"] == "report.html"
    assert issue["browse_url"].endswith("/browse/CURR-42")


def test_qa_report_issue_detection_prefers_direct_sources_and_report_language():
    assert is_qa_report_issue(
        {
            "summary": "Curriculum work",
            "description": "",
            "links": [
                "https://github.com/ayo-kiddom/content-enhancement-agent/"
                "actions/runs/123"
            ],
            "attachments": [],
        }
    )
    assert is_qa_report_issue(
        {
            "summary": "Grade 2 QA Report Review",
            "description": "",
            "links": [],
            "attachments": [],
        }
    )
    assert not is_qa_report_issue(
        {
            "summary": "Update curriculum metadata",
            "description": "Routine ticket",
            "links": ["https://example.test"],
            "attachments": [],
        }
    )


def test_completed_handoff_attaches_transitions_and_reassigns():
    session = FakeSession(
        [
            FakeResponse(data=raw_issue()),
            FakeResponse(data=[{"id": "9002", "filename": "result.csv"}]),
            FakeResponse(status_code=204),
            FakeResponse(
                data={
                    "transitions": [
                        {
                            "id": "31",
                            "name": "Send to QA",
                            "to": {"name": "Ready for QA"},
                        }
                    ]
                }
            ),
            FakeResponse(status_code=204),
        ]
    )
    client = JiraClient(config(), session=session)
    steps = client.handoff_completed_review(
        "CURR-42",
        "result.csv",
        b'"issue_id","status","comment"\r\n',
        "qa-123",
        "Ready for QA",
    )

    assert [step["result"] for step in steps] == [
        "Attached",
        "Reassigned",
        "Moved to Ready for QA",
    ]
    attachment_call = session.calls[1]
    assert attachment_call[2]["headers"]["X-Atlassian-Token"] == "no-check"
    assert attachment_call[2]["files"]["file"][0] == "result.csv"
    assert session.calls[2][2]["json"] == {"accountId": "qa-123"}
    assert session.calls[4][2]["json"] == {"transition": {"id": "31"}}


def test_completed_handoff_comments_all_findings_csv_before_reassignment():
    session = FakeSession(
        [
            FakeResponse(data=raw_issue()),
            FakeResponse(
                data=[
                    {
                        "id": "9002",
                        "filename": "result.csv",
                        "content": "https://example.atlassian.net/attachment/9002",
                    }
                ]
            ),
            FakeResponse(data={"comments": []}),
            FakeResponse(status_code=201, data={"id": "7002"}),
            FakeResponse(status_code=204),
            FakeResponse(
                data={
                    "transitions": [
                        {
                            "id": "31",
                            "name": "Send to QA",
                            "to": {"name": "Ready for QA"},
                        }
                    ]
                }
            ),
            FakeResponse(status_code=204),
        ]
    )
    steps = JiraClient(config(), session=session).handoff_completed_review(
        "CURR-42",
        "result.csv",
        b"csv",
        "qa-123",
        "Ready for QA",
        comment_text="All 10 findings are included.",
        comment_marker="[KIDDOM-QA-FINAL:abc123]",
    )
    assert [step["result"] for step in steps] == [
        "Attached",
        "Commented",
        "Reassigned",
        "Moved to Ready for QA",
    ]
    assert session.calls[3][0] == "POST"
    assert session.calls[3][1].endswith("/issue/CURR-42/comment")
    assert "All 10 findings" in str(session.calls[3][2]["json"])
    assert session.calls[4][1].endswith("/issue/CURR-42/assignee")
    assert session.calls[6][1].endswith("/issue/CURR-42/transitions")


def test_transition_by_id_uses_current_jira_options():
    session = FakeSession(
        [
            FakeResponse(
                data={
                    "transitions": [
                        {
                            "id": "41",
                            "name": "Start work",
                            "to": {"name": "In Progress"},
                        }
                    ]
                }
            ),
            FakeResponse(status_code=204),
        ]
    )
    client = JiraClient(config(), session=session)

    result = client.transition_issue_by_id("CURR-42", "41")

    assert result == "In Progress"
    assert session.calls[1][2]["json"] == {"transition": {"id": "41"}}


def test_handoff_retry_skips_already_completed_actions():
    session = FakeSession(
        [
            FakeResponse(
                data=raw_issue(
                    status="Ready for QA",
                    assignee="qa-123",
                    attachments=[
                        {
                            "id": "9002",
                            "filename": "result.csv",
                            "mimeType": "text/csv",
                        }
                    ],
                )
            )
        ]
    )
    steps = JiraClient(config(), session=session).handoff_completed_review(
        "CURR-42",
        "result.csv",
        b"same",
        "qa-123",
        "Ready for QA",
    )
    assert [step["result"] for step in steps] == [
        "Already attached",
        "Already assigned",
        "Already Ready for QA",
    ]
    assert len(session.calls) == 1


def test_partial_handoff_reports_completed_steps():
    session = FakeSession(
        [
            FakeResponse(data=raw_issue()),
            FakeResponse(data=[{"id": "9002"}]),
            FakeResponse(status_code=403, data={"errorMessages": ["No permission"]}),
        ]
    )
    client = JiraClient(config(), session=session)
    with pytest.raises(JiraHandoffError) as caught:
        client.handoff_completed_review(
            "CURR-42",
            "result.csv",
            b"csv",
            "qa-123",
            "Ready for QA",
        )
    assert caught.value.completed_steps == [
        {"step": "CSV attachment", "result": "Attached"}
    ]
    assert "No permission" in str(caught.value)


def test_source_report_attachment_and_comment_are_retry_safe():
    marker = "[KIDDOM-QA-SOURCE:abc123]"
    session = FakeSession(
        [
            FakeResponse(data=raw_issue()),
            FakeResponse(
                data=[
                    {
                        "id": "9003",
                        "filename": "source.html",
                        "content": "https://example.atlassian.net/attachment/9003",
                    }
                ]
            ),
            FakeResponse(data={"comments": []}),
            FakeResponse(status_code=201, data={"id": "7001"}),
        ]
    )
    steps = JiraClient(config(), session=session).ensure_file_comment(
        "CURR-42",
        "source.html",
        b"<html>report</html>",
        mime_type="text/html",
        comment_text="Source from https://github.com/example/run",
        comment_marker=marker,
        attachment_step="Source HTML",
        comment_step="Source comment",
    )
    assert [step["result"] for step in steps] == ["Attached", "Commented"]
    comment_payload = session.calls[3][2]["json"]
    assert comment_payload["body"]["type"] == "doc"
    assert marker in str(comment_payload)
    assert '"type": "link"' in str(comment_payload).replace("'", '"')

    retry_session = FakeSession(
        [
            FakeResponse(
                data=raw_issue(
                    attachments=[
                        {
                            "id": "9003",
                            "filename": "source.html",
                            "mimeType": "text/html",
                        }
                    ]
                )
            ),
            FakeResponse(
                data={
                    "comments": [
                        {
                            "id": "7001",
                            "body": {
                                "type": "doc",
                                "content": [
                                    {
                                        "type": "paragraph",
                                        "content": [
                                            {"type": "text", "text": marker}
                                        ],
                                    }
                                ],
                            },
                        }
                    ]
                }
            ),
        ]
    )
    retry_steps = JiraClient(
        config(),
        session=retry_session,
    ).ensure_file_comment(
        "CURR-42",
        "source.html",
        b"<html>report</html>",
        mime_type="text/html",
        comment_text="Source",
        comment_marker=marker,
        attachment_step="Source HTML",
        comment_step="Source comment",
    )
    assert [step["result"] for step in retry_steps] == [
        "Already attached",
        "Already commented",
    ]
    assert len(retry_session.calls) == 2
