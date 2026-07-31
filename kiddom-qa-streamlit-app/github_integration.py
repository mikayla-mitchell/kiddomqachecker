from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping
from urllib.parse import quote, urlparse

import requests


GITHUB_API_VERSION = "2026-03-10"
RUN_PATH_RE = re.compile(
    r"^/([^/]+)/([^/]+)/actions/runs/(\d+)(?:/.*)?$",
    re.I,
)
REPORT_MARKERS = (b"issue-card", b"aggregated-issue")


class GitHubIntegrationError(RuntimeError):
    """A safe, user-facing GitHub integration error."""


@dataclass(frozen=True)
class GitHubConfig:
    token: str
    api_url: str = "https://api.github.com"
    max_artifact_bytes: int = 400 * 1024 * 1024
    max_report_bytes: int = 1_200 * 1024 * 1024

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "GitHubConfig":
        token = str(values.get("GITHUB_TOKEN") or "").strip()
        if not token:
            raise GitHubIntegrationError(
                "GitHub report loading is not configured. Missing GITHUB_TOKEN."
            )
        api_url = str(
            values.get("GITHUB_API_URL") or "https://api.github.com"
        ).strip().rstrip("/")
        parsed = urlparse(api_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise GitHubIntegrationError(
                "GITHUB_API_URL must be a complete HTTPS URL."
            )
        if parsed.query or parsed.fragment:
            raise GitHubIntegrationError(
                "GITHUB_API_URL cannot include a query string or fragment."
            )

        def byte_limit(name: str, default: int) -> int:
            raw = values.get(name)
            if raw in (None, ""):
                return default
            try:
                limit = int(raw)
            except (TypeError, ValueError) as error:
                raise GitHubIntegrationError(
                    f"{name} must be a whole number of bytes."
                ) from error
            if not 1 <= limit <= 1_200 * 1024 * 1024:
                raise GitHubIntegrationError(
                    f"{name} must be between 1 byte and 1.2 GB."
                )
            return limit

        return cls(
            token=token,
            api_url=api_url,
            max_artifact_bytes=byte_limit(
                "GITHUB_MAX_ARTIFACT_BYTES", 400 * 1024 * 1024
            ),
            max_report_bytes=byte_limit(
                "GITHUB_MAX_REPORT_BYTES", 1_200 * 1024 * 1024
            ),
        )


@dataclass(frozen=True)
class GitHubRunRef:
    owner: str
    repo: str
    run_id: str
    web_url: str

    @property
    def label(self) -> str:
        return f"{self.owner}/{self.repo} · run {self.run_id}"

def parse_github_actions_run_url(url: str) -> GitHubRunRef | None:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme != "https" or parsed.hostname not in {
        "github.com",
        "www.github.com",
    }:
        return None
    match = RUN_PATH_RE.match(parsed.path.rstrip("/"))
    if not match:
        return None
    owner, repo, run_id = match.groups()
    return GitHubRunRef(
        owner=owner,
        repo=repo.removesuffix(".git"),
        run_id=run_id,
        web_url=f"https://github.com/{owner}/{repo.removesuffix('.git')}/actions/runs/{run_id}",
    )


def github_run_refs(links: list[str]) -> list[GitHubRunRef]:
    refs: list[GitHubRunRef] = []
    seen: set[tuple[str, str, str]] = set()
    for link in links:
        ref = parse_github_actions_run_url(link)
        if ref is None:
            continue
        key = (ref.owner.casefold(), ref.repo.casefold(), ref.run_id)
        if key not in seen:
            seen.add(key)
            refs.append(ref)
    return refs


class GitHubActionsClient:
    def __init__(
        self,
        config: GitHubConfig,
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
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.config.token}",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            **kwargs.pop("headers", {}),
        }
        try:
            response = self.session.request(
                method,
                f"{self.config.api_url}{path}",
                headers=headers,
                timeout=(10, 120),
                **kwargs,
            )
        except requests.RequestException as error:
            raise GitHubIntegrationError(
                f"GitHub could not be reached: {error.__class__.__name__}."
            ) from error
        if response.status_code not in expected:
            if response.status_code == 401:
                raise GitHubIntegrationError(
                    "GitHub rejected the token saved in Streamlit. Replace "
                    "the [github] token secret, save, and reboot the app."
                )
            if response.status_code == 403:
                raise GitHubIntegrationError(
                    "GitHub accepted the token but denied this request. "
                    "Confirm it has Actions: read access to the repository "
                    "named above."
                )
            if response.status_code == 404:
                raise GitHubIntegrationError(
                    "GitHub could not find this workflow run using the saved "
                    "token. The run may still exist but be hidden because "
                    "the token is scoped to a different repository owner."
                )
            if response.status_code == 410:
                raise GitHubIntegrationError(
                    "This GitHub Actions artifact has expired."
                )
            raise GitHubIntegrationError(
                f"GitHub returned HTTP {response.status_code}."
            )
        return response

    def list_run_artifacts(self, ref: GitHubRunRef) -> list[dict[str, Any]]:
        path = (
            f"/repos/{quote(ref.owner, safe='')}/{quote(ref.repo, safe='')}"
            f"/actions/runs/{quote(ref.run_id, safe='')}/artifacts"
        )
        response = self._request(
            "GET",
            path,
            params={"per_page": 100},
        )
        try:
            payload = response.json()
        except ValueError as error:
            raise GitHubIntegrationError(
                "GitHub returned an unreadable artifact list."
            ) from error
        raw_artifacts = payload.get("artifacts") if isinstance(payload, Mapping) else []
        if not isinstance(raw_artifacts, list):
            raise GitHubIntegrationError(
                "GitHub returned an unreadable artifact list."
            )
        artifacts = []
        for raw in raw_artifacts:
            if not isinstance(raw, Mapping):
                continue
            artifacts.append(
                {
                    "id": str(raw.get("id") or ""),
                    "name": str(raw.get("name") or "artifact"),
                    "size": int(raw.get("size_in_bytes") or 0),
                    "expired": bool(raw.get("expired")),
                }
            )
        return [artifact for artifact in artifacts if artifact["id"]]

    def download_artifact(self, ref: GitHubRunRef, artifact_id: str) -> bytes:
        path = (
            f"/repos/{quote(ref.owner, safe='')}/{quote(ref.repo, safe='')}"
            f"/actions/artifacts/{quote(str(artifact_id), safe='')}/zip"
        )
        response = self._request(
            "GET",
            path,
            expected=(200,),
            allow_redirects=True,
            stream=True,
        )
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > self.config.max_artifact_bytes:
                    raise GitHubIntegrationError(
                        "The GitHub artifact is larger than the configured "
                        "in-app download limit."
                    )
            except ValueError:
                pass

        payload = bytearray()
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            payload.extend(chunk)
            if len(payload) > self.config.max_artifact_bytes:
                raise GitHubIntegrationError(
                    "The GitHub artifact is larger than the configured "
                    "in-app download limit."
                )
        return bytes(payload)

    def report_files_from_artifact(
        self,
        archive_payload: bytes,
        artifact: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            archive = zipfile.ZipFile(io.BytesIO(archive_payload))
        except zipfile.BadZipFile as error:
            raise GitHubIntegrationError(
                "GitHub returned an invalid artifact ZIP file."
            ) from error

        reports: list[dict[str, Any]] = []
        retained_bytes = 0
        try:
            with archive:
                for info in archive.infolist():
                    if info.is_dir() or info.flag_bits & 0x1:
                        continue
                    path = PurePosixPath(info.filename)
                    if path.suffix.casefold() not in {".html", ".htm"}:
                        continue
                    if info.file_size > self.config.max_report_bytes:
                        raise GitHubIntegrationError(
                            f"{path.name} is larger than the configured report limit."
                        )
                    with archive.open(info) as source:
                        payload = source.read(self.config.max_report_bytes + 1)
                    if len(payload) > self.config.max_report_bytes:
                        raise GitHubIntegrationError(
                            f"{path.name} is larger than the configured report limit."
                        )
                    lowered = payload.lower()
                    if not all(marker in lowered for marker in REPORT_MARKERS):
                        continue
                    retained_bytes += len(payload)
                    if retained_bytes > self.config.max_report_bytes:
                        raise GitHubIntegrationError(
                            "The reports in this artifact exceed the configured "
                            "in-app report limit."
                        )
                    reports.append(
                        {
                            "filename": path.name,
                            "archive_path": str(path),
                            "payload": payload,
                            "artifact_id": str(artifact.get("id") or ""),
                            "artifact_name": str(
                                artifact.get("name") or "GitHub artifact"
                            ),
                        }
                    )
        except (zipfile.BadZipFile, RuntimeError) as error:
            raise GitHubIntegrationError(
                "GitHub returned an unreadable artifact ZIP file."
            ) from error
        return reports

    def find_report_files(
        self,
        ref: GitHubRunRef,
        *,
        max_artifacts: int = 10,
    ) -> list[dict[str, Any]]:
        artifacts = self.list_run_artifacts(ref)
        available = [artifact for artifact in artifacts if not artifact["expired"]]
        if not available:
            if artifacts:
                raise GitHubIntegrationError(
                    "All artifacts for this workflow run have expired."
                )
            raise GitHubIntegrationError(
                "This workflow run does not contain any artifacts yet."
            )

        keywords = ("qa", "report", "annotat", "review")
        available.sort(
            key=lambda artifact: (
                0
                if any(word in artifact["name"].casefold() for word in keywords)
                else 1,
                artifact["size"],
                artifact["name"].casefold(),
            )
        )
        checked = 0
        for artifact in available:
            if checked >= max_artifacts:
                break
            if artifact["size"] > self.config.max_artifact_bytes:
                continue
            checked += 1
            archive_payload = self.download_artifact(ref, artifact["id"])
            reports = self.report_files_from_artifact(archive_payload, artifact)
            if reports:
                return reports
        raise GitHubIntegrationError(
            "No Issue Annotation Report HTML file was found in the available "
            "workflow artifacts."
        )
