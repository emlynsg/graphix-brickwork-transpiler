"""Nox sessions for linting, typing, and testing."""

from __future__ import annotations

import nox

nox.options.sessions = ["lint", "typecheck", "test"]
nox.options.reuse_existing_virtualenvs = True


@nox.session(python="3.12")
def lint(session: nox.Session) -> None:
    """Run Ruff steps."""
    session.install("ruff")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session(python="3.12")
def typecheck(session: nox.Session) -> None:
    """Run Mypy checks."""
    session.install("mypy", ".")
    session.run("mypy", ".", "--ignore-missing-imports")


@nox.session(python="3.12")
def test(session: nox.Session) -> None:
    """Run Pytest checks."""
    session.install("pytest", ".")
    session.run("pytest", "-q")
