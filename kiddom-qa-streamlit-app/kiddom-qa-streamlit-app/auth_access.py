from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


class AccessDeniedError(ValueError):
    """An authenticated identity is not allowed to use the app."""


@dataclass(frozen=True)
class UserIdentity:
    name: str
    email: str
    picture: str = ""
    is_authenticated: bool = True
    is_admin: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_email_values(values: Iterable[Any] | str | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = [values]
    return tuple(
        dict.fromkeys(
            str(value).strip().casefold()
            for value in values
            if str(value).strip()
        )
    )


def build_user_identity(
    claims: Mapping[str, Any],
    *,
    allowed_email_domains: Iterable[Any] | str | None = None,
    admin_emails: Iterable[Any] | str | None = None,
) -> UserIdentity:
    email = str(claims.get("email") or "").strip().casefold()
    if not email or "@" not in email:
        raise AccessDeniedError(
            "Google did not provide a valid email address for this account."
        )

    verified = claims.get("email_verified", True)
    if verified is False or str(verified).strip().casefold() == "false":
        raise AccessDeniedError("Your Google Workspace email is not verified.")

    allowed_domains = normalize_email_values(allowed_email_domains)
    domain = email.rsplit("@", 1)[1]
    if allowed_domains and domain not in allowed_domains:
        raise AccessDeniedError(
            "This app is restricted to approved Google Workspace accounts."
        )

    hosted_domain = str(claims.get("hd") or "").strip().casefold()
    if hosted_domain and allowed_domains and hosted_domain not in allowed_domains:
        raise AccessDeniedError(
            "This Google account does not belong to an approved Workspace domain."
        )

    admins = normalize_email_values(admin_emails)
    return UserIdentity(
        name=str(claims.get("name") or email.split("@", 1)[0]).strip(),
        email=email,
        picture=str(claims.get("picture") or "").strip(),
        is_authenticated=True,
        is_admin=email in admins,
    )


def local_development_identity() -> UserIdentity:
    return UserIdentity(
        name="Local development",
        email="local@localhost",
        is_authenticated=False,
        is_admin=True,
    )
