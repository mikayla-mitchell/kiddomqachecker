from __future__ import annotations

import io
import zipfile

import pytest

from github_integration import (
    GitHubActionsClient,
    GitHubConfig,
    GitHubIntegrationError,
    github_run_refs,
    parse_github_actions_run_url,
)


class FakeResponse:
    def __init__(
        self,
        status_code=200,
        *,
        payload=None,
        content=b"",
        headers=None,
    ):
        self.status_code = status_code
        self._payload = payload
        self._content = content
        self.headers = headers or {}

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def iter_content(self, chunk_size=1024):
        del chunk_size
        yield self._content


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def artifact_zip(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return stream.getvalue()


def report_html(label="Report") -> bytes:
    return (
        f'<html><title>{label}</title><div class="issue-card">'
        '<div class="aggregated-issue"></div></div></html>'
    ).encode()


def test_parse_github_actions_run_url_and_deduplicate():
    ref = parse_github_actions_run_url(
        "https://github.com/kiddom/curriculum/actions/runs/28899103438/job/1"
    )
    assert ref is not None
    assert (ref.owner, ref.repo, ref.run_id) == (
        "kiddom",
        "curriculum",
        "28899103438",
    )
    assert (
        parse_github_actions_run_url(
            "https://github.com/kiddom/curriculum/blob/main/report.html"
        )
        is None
    )
    refs = github_run_refs(
        [
            ref.web_url,
            ref.web_url + "?check_suite_focus=true",
            "https://example.test/not-github",
        ]
    )
    assert refs == [ref]


def test_config_requires_server_side_token():
    with pytest.raises(GitHubIntegrationError, match="GITHUB_TOKEN"):
        GitHubConfig.from_mapping({})
    config = GitHubConfig.from_mapping({"GITHUB_TOKEN": "secret"})
    assert config.api_url == "https://api.github.com"


def test_list_and_download_artifact_uses_read_only_api_headers():
    session = FakeSession(
        [
            FakeResponse(
                payload={
                    "artifacts": [
                        {
                            "id": 42,
                            "name": "qa-report",
                            "size_in_bytes": 1200,
                            "expired": False,
                        }
                    ]
                }
            ),
            FakeResponse(
                content=artifact_zip({"course/report.html": report_html()}),
            ),
        ]
    )
    config = GitHubConfig.from_mapping({"GITHUB_TOKEN": "secret"})
    client = GitHubActionsClient(config, session=session)
    ref = parse_github_actions_run_url(
        "https://github.com/kiddom/curriculum/actions/runs/123"
    )
    assert ref is not None

    artifacts = client.list_run_artifacts(ref)
    assert artifacts[0]["name"] == "qa-report"
    archive = client.download_artifact(ref, "42")
    reports = client.report_files_from_artifact(archive, artifacts[0])
    assert reports[0]["filename"] == "report.html"
    assert reports[0]["artifact_name"] == "qa-report"

    list_call, download_call = session.calls
    assert list_call[1].endswith(
        "/repos/kiddom/curriculum/actions/runs/123/artifacts"
    )
    assert download_call[1].endswith(
        "/repos/kiddom/curriculum/actions/artifacts/42/zip"
    )
    assert list_call[2]["headers"]["Authorization"] == "Bearer secret"
    assert download_call[2]["allow_redirects"] is True


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (401, "rejected the token saved in Streamlit"),
        (403, "accepted the token but denied"),
        (404, "could not find this workflow run"),
    ],
)
def test_github_auth_errors_explain_the_next_step(status_code, message):
    client = GitHubActionsClient(
        GitHubConfig.from_mapping({"GITHUB_TOKEN": "secret"}),
        session=FakeSession([FakeResponse(status_code=status_code)]),
    )
    ref = parse_github_actions_run_url(
        "https://github.com/kiddom/content-enhancement-agent/actions/runs/123"
    )
    assert ref is not None

    with pytest.raises(GitHubIntegrationError, match=message):
        client.list_run_artifacts(ref)


def test_find_report_files_prefers_report_named_artifact():
    unrelated_zip = artifact_zip({"results.txt": b"ok"})
    report_zip = artifact_zip(
        {
            "course/summary.html": b"<html>not a report</html>",
            "course/review_detailed.html": report_html("Detailed"),
        }
    )
    session = FakeSession(
        [
            FakeResponse(
                payload={
                    "artifacts": [
                        {
                            "id": 1,
                            "name": "build-output",
                            "size_in_bytes": len(unrelated_zip),
                            "expired": False,
                        },
                        {
                            "id": 2,
                            "name": "qa-reports",
                            "size_in_bytes": len(report_zip),
                            "expired": False,
                        },
                    ]
                }
            ),
            FakeResponse(content=report_zip),
        ]
    )
    client = GitHubActionsClient(
        GitHubConfig.from_mapping({"GITHUB_TOKEN": "secret"}),
        session=session,
    )
    ref = parse_github_actions_run_url(
        "https://github.com/kiddom/curriculum/actions/runs/456"
    )
    assert ref is not None
    reports = client.find_report_files(ref)
    assert [report["filename"] for report in reports] == [
        "review_detailed.html"
    ]
    assert session.calls[1][1].endswith("/artifacts/2/zip")


def test_expired_or_missing_report_artifacts_get_clear_errors():
    ref = parse_github_actions_run_url(
        "https://github.com/kiddom/curriculum/actions/runs/789"
    )
    assert ref is not None
    expired_client = GitHubActionsClient(
        GitHubConfig.from_mapping({"GITHUB_TOKEN": "secret"}),
        session=FakeSession(
            [
                FakeResponse(
                    payload={
                        "artifacts": [
                            {
                                "id": 1,
                                "name": "qa-report",
                                "size_in_bytes": 1,
                                "expired": True,
                            }
                        ]
                    }
                )
            ]
        ),
    )
    with pytest.raises(GitHubIntegrationError, match="expired"):
        expired_client.find_report_files(ref)

    missing_client = GitHubActionsClient(
        GitHubConfig.from_mapping({"GITHUB_TOKEN": "secret"}),
        session=FakeSession(
            [
                FakeResponse(
                    payload={
                        "artifacts": [
                            {
                                "id": 2,
                                "name": "qa-report",
                                "size_in_bytes": 20,
                                "expired": False,
                            }
                        ]
                    }
                ),
                FakeResponse(content=artifact_zip({"notes.html": b"<html></html>"})),
            ]
        ),
    )
    with pytest.raises(GitHubIntegrationError, match="No Issue Annotation"):
        missing_client.find_report_files(ref)
