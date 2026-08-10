#!/usr/bin/env python3
"""Synchronize the default branches of repositories forked by a GitHub user."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

API_VERSION = "2022-11-28"
DEFAULT_API_URL = "https://api.github.com"
PER_PAGE = 100

JsonObject: TypeAlias = dict[str, object]
QueryParameters: TypeAlias = dict[str, str | int]


class ConfigurationError(RuntimeError):
    """Raised when the synchronizer configuration is invalid."""


class GitHubApiError(RuntimeError):
    """An error returned while calling the GitHub API."""

    status: int | None

    def __init__(self, status: int | None, message: str) -> None:
        self.status = status
        prefix = f"HTTP {status}" if status is not None else "GitHub API error"
        super().__init__(f"{prefix}: {message}")


class GitHubClient:
    token: str
    api_url: str
    timeout: float

    def __init__(
        self,
        token: str,
        api_url: str = DEFAULT_API_URL,
        timeout: float = 30,
    ) -> None:
        self.token = token
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: QueryParameters | None = None,
        payload: JsonObject | None = None,
    ) -> object:
        url = f"{self.api_url}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{urlencode(query)}"

        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "reposync-github-action",
            "X-GitHub-Api-Version": API_VERSION,
        }
        if data is not None:
            headers["Content-Type"] = "application/json"

        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body: bytes = response.read()
                status: int = response.status
        except HTTPError as error:
            body = error.read()
            message = (
                _github_error_message(body) or str(error.reason) or "request failed"
            )
            raise GitHubApiError(
                error.code, f"{method} {path}: {message}"
            ) from error
        except URLError as error:
            raise GitHubApiError(
                None, f"{method} {path}: network error: {error.reason}"
            ) from error
        except TimeoutError as error:
            raise GitHubApiError(
                None, f"{method} {path}: request timed out after {self.timeout:g}s"
            ) from error

        if not body:
            return None

        try:
            return cast(object, json.loads(body.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GitHubApiError(
                status, f"{method} {path}: GitHub returned invalid JSON"
            ) from error

    def authenticated_user(self) -> JsonObject:
        response = self._request("GET", "/user")
        if not isinstance(response, dict):
            raise GitHubApiError(None, "GET /user returned an unexpected response")
        return cast(JsonObject, response)

    def owned_repositories(self) -> list[JsonObject]:
        repositories: list[JsonObject] = []
        page = 1

        while True:
            response = self._request(
                "GET",
                "/user/repos",
                query={
                    "affiliation": "owner",
                    "visibility": "all",
                    "sort": "full_name",
                    "direction": "asc",
                    "per_page": PER_PAGE,
                    "page": page,
                },
            )
            if not isinstance(response, list):
                raise GitHubApiError(
                    None, "GET /user/repos returned an unexpected response"
                )

            response_items = cast(list[object], response)
            repositories.extend(
                cast(JsonObject, repository)
                for repository in response_items
                if isinstance(repository, dict)
            )
            if len(response_items) < PER_PAGE:
                return repositories
            page += 1

    def merge_upstream(
        self, owner: str, repository: str, branch: str
    ) -> JsonObject:
        response = self._request(
            "POST",
            f"/repos/{quote(owner, safe='')}/{quote(repository, safe='')}/merge-upstream",
            payload={"branch": branch},
        )
        if not isinstance(response, dict):
            raise GitHubApiError(
                None, "merge-upstream returned an unexpected response"
            )
        return cast(JsonObject, response)


class SyncClient(Protocol):
    def authenticated_user(self) -> JsonObject: ...

    def owned_repositories(self) -> list[JsonObject]: ...

    def merge_upstream(
        self, owner: str, repository: str, branch: str
    ) -> JsonObject: ...


@dataclass(frozen=True)
class Outcome:
    repository: str
    detail: str


@dataclass
class SyncSummary:
    repositories_seen: int = 0
    forks_found: int = 0
    successful: list[Outcome] = field(default_factory=list)
    planned: list[Outcome] = field(default_factory=list)
    skipped: list[Outcome] = field(default_factory=list)
    failed: list[Outcome] = field(default_factory=list)


def synchronize(
    client: SyncClient,
    expected_owner: str | None,
    *,
    dry_run: bool = False,
    emit: Callable[[str], None] = print,
) -> SyncSummary:
    user = client.authenticated_user()
    authenticated_owner = user.get("login")
    if not isinstance(authenticated_owner, str) or not authenticated_owner:
        raise ConfigurationError("GitHub did not return the authenticated user's login")

    owner = expected_owner or authenticated_owner
    if owner.casefold() != authenticated_owner.casefold():
        message = (
            f"FORK_SYNC_TOKEN belongs to {authenticated_owner!r}, but SYNC_OWNER "
            f"is {owner!r}"
        )
        raise ConfigurationError(message)

    repositories = client.owned_repositories()
    summary = SyncSummary(repositories_seen=len(repositories))
    forks = [repository for repository in repositories if repository.get("fork") is True]
    summary.forks_found = len(forks)

    discovery_message = (
        f"Authenticated as {authenticated_owner}. Found {len(forks)} fork(s) among "
        f"{len(repositories)} owned repository/repositories."
    )
    emit(discovery_message)

    for repository in forks:
        repository_owner = _repository_owner(repository) or authenticated_owner
        repository_name = repository.get("name")
        full_name = repository.get("full_name")
        if not isinstance(repository_name, str) or not repository_name:
            summary.skipped.append(Outcome(str(full_name or "<unknown>"), "missing name"))
            continue
        if not isinstance(full_name, str) or not full_name:
            full_name = f"{repository_owner}/{repository_name}"

        if repository_owner.casefold() != authenticated_owner.casefold():
            detail = f"repository owner is {repository_owner!r}, not {authenticated_owner!r}"
            summary.skipped.append(Outcome(full_name, detail))
            emit(f"[SKIP] {full_name}: {detail}")
            continue
        if repository.get("archived") is True:
            detail = "repository is archived"
            summary.skipped.append(Outcome(full_name, detail))
            emit(f"[SKIP] {full_name}: {detail}")
            continue
        if repository.get("disabled") is True:
            detail = "repository is disabled"
            summary.skipped.append(Outcome(full_name, detail))
            emit(f"[SKIP] {full_name}: {detail}")
            continue

        branch = repository.get("default_branch")
        if not isinstance(branch, str) or not branch:
            detail = "repository has no default branch"
            summary.skipped.append(Outcome(full_name, detail))
            emit(f"[SKIP] {full_name}: {detail}")
            continue

        if dry_run:
            detail = f"would synchronize default branch {branch!r}"
            summary.planned.append(Outcome(full_name, detail))
            emit(f"[DRY RUN] {full_name}: {detail}")
            continue

        try:
            response = client.merge_upstream(repository_owner, repository_name, branch)
        except GitHubApiError as error:
            if error.status == 409:
                detail = f"upstream merge conflict: {error}"
            else:
                detail = str(error)
            summary.failed.append(Outcome(full_name, detail))
            emit(f"[ERROR] {full_name}: {detail}")
            continue

        detail = _success_message(response)
        summary.successful.append(Outcome(full_name, detail))
        emit(f"[OK] {full_name}: {detail}")

    return summary


def _repository_owner(repository: JsonObject) -> str | None:
    owner = repository.get("owner")
    if isinstance(owner, dict):
        owner_data = cast(JsonObject, owner)
        login = owner_data.get("login")
        if isinstance(login, str) and login:
            return login
    return None


def _success_message(response: JsonObject) -> str:
    message = response.get("message")
    merge_type = response.get("merge_type")
    detail = message if isinstance(message, str) and message else "synchronization accepted"
    if isinstance(merge_type, str) and merge_type:
        detail = f"{detail} (merge type: {merge_type})"
    return detail


def _github_error_message(body: bytes) -> str | None:
    if not body:
        return None
    try:
        payload = cast(object, json.loads(body.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body.decode("utf-8", errors="replace").strip() or None

    if not isinstance(payload, dict):
        return str(payload)

    payload_data = cast(JsonObject, payload)
    message = payload_data.get("message")
    errors = payload_data.get("errors")
    if errors:
        serialized_errors = json.dumps(errors, ensure_ascii=False, separators=(",", ":"))
        return f"{message or 'request failed'}; errors={serialized_errors}"
    return str(message) if message else None


def _read_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false, got {value!r}")


def _write_step_summary(summary: SyncSummary, dry_run: bool) -> None:
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    successful_label = "Planned" if dry_run else "Successful"
    successful_count = len(summary.planned) if dry_run else len(summary.successful)
    lines = [
        "## Fork synchronization",
        "",
        f"- Owned repositories scanned: **{summary.repositories_seen}**",
        f"- Forks found: **{summary.forks_found}**",
        f"- {successful_label}: **{successful_count}**",
        f"- Skipped: **{len(summary.skipped)}**",
        f"- Failed: **{len(summary.failed)}**",
    ]

    details = summary.planned if dry_run else summary.successful
    if details:
        lines.extend(["", f"### {successful_label}"])
        lines.extend(
            f"- `{outcome.repository}` — {_markdown(outcome.detail)}"
            for outcome in details
        )
    if summary.skipped:
        lines.extend(["", "### Skipped"])
        lines.extend(
            f"- `{outcome.repository}` — {_markdown(outcome.detail)}"
            for outcome in summary.skipped
        )
    if summary.failed:
        lines.extend(["", "### Failed"])
        lines.extend(
            f"- `{outcome.repository}` — {_markdown(outcome.detail)}"
            for outcome in summary.failed
        )

    try:
        with open(summary_path, "a", encoding="utf-8") as summary_file:
            _ = summary_file.write("\n".join(lines) + "\n")
    except OSError as error:
        print(f"Could not write GITHUB_STEP_SUMMARY: {error}", file=sys.stderr)


def _markdown(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", " ").replace("|", "\\|")


class Arguments(argparse.Namespace):
    owner: str | None
    api_url: str
    timeout: float
    dry_run: bool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--owner",
        default=os.getenv("SYNC_OWNER"),
        help="expected GitHub login (defaults to SYNC_OWNER or the token owner)",
    )
    _ = parser.add_argument(
        "--api-url",
        default=os.getenv("GITHUB_API_URL", DEFAULT_API_URL),
        help="GitHub REST API base URL",
    )
    _ = parser.add_argument(
        "--timeout",
        type=float,
        default=30,
        help="per-request timeout in seconds (default: 30)",
    )
    _ = parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list forks without calling merge-upstream",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv, namespace=Arguments())

    try:
        dry_run = args.dry_run or _read_bool_env("DRY_RUN")
        token = os.getenv("FORK_SYNC_TOKEN", "").strip()
        if not token:
            raise ConfigurationError(
                "FORK_SYNC_TOKEN is empty; add it as a GitHub Actions secret"
            )
        if args.timeout <= 0:
            raise ConfigurationError("--timeout must be greater than zero")

        client = GitHubClient(token, api_url=args.api_url, timeout=args.timeout)
        summary = synchronize(client, args.owner, dry_run=dry_run)
    except (ConfigurationError, GitHubApiError) as error:
        print(f"Fatal error: {error}", file=sys.stderr)
        return 2

    _write_step_summary(summary, dry_run)
    if summary.failed:
        failure_message = (
            f"Completed with {len(summary.failed)} failure(s); all other eligible "
            "forks were still processed."
        )
        print(failure_message, file=sys.stderr)
        return 1

    action = "planned" if dry_run else "completed"
    successful_count = len(summary.planned) if dry_run else len(summary.successful)
    print(
        f"Synchronization {action}: {successful_count} successful, "
        f"{len(summary.skipped)} skipped."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
