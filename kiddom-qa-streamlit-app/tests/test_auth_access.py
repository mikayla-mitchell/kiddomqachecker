from __future__ import annotations

import pytest

from auth_access import AccessDeniedError, build_user_identity


def test_workspace_identity_enforces_domain_and_marks_admin():
    identity = build_user_identity(
        {
            "email": "Mikayla@Kiddom.co",
            "name": "Mikayla Mitchell",
            "email_verified": True,
            "hd": "kiddom.co",
        },
        allowed_email_domains=["kiddom.co"],
        admin_emails=["mikayla@kiddom.co"],
    )
    assert identity.email == "mikayla@kiddom.co"
    assert identity.name == "Mikayla Mitchell"
    assert identity.is_admin is True


def test_workspace_identity_rejects_unapproved_or_unverified_accounts():
    with pytest.raises(AccessDeniedError, match="approved"):
        build_user_identity(
            {"email": "person@gmail.com", "email_verified": True},
            allowed_email_domains=["kiddom.co"],
        )
    with pytest.raises(AccessDeniedError, match="not verified"):
        build_user_identity(
            {"email": "person@kiddom.co", "email_verified": False},
            allowed_email_domains=["kiddom.co"],
        )
