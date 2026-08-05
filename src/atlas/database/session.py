"""Database engine and session management.

SQLite is the local default; any PostgreSQL-compatible URL works unchanged
because the models avoid backend-specific types. The URL comes from
``ATLAS_DATABASE_URL`` when set, otherwise a file under ``data/database/``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from atlas.database.models import Base
from atlas.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["DatabaseManager", "default_database_url"]


def default_database_url(project_root: Path | None = None) -> str:
    """Return the configured database URL.

    Order of precedence: ``ATLAS_DATABASE_URL``, then a SQLite file at
    ``<project_root>/data/database/atlas.db``.
    """
    configured = os.environ.get("ATLAS_DATABASE_URL")
    if configured:
        return configured
    root = project_root or Path.cwd()
    path = Path(root) / "data" / "database" / "atlas.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


class DatabaseManager:
    """Own the engine and hand out sessions.

    Parameters
    ----------
    url:
        SQLAlchemy database URL. Defaults to :func:`default_database_url`.
    echo:
        Log every statement (useful when debugging, very noisy otherwise).
    create_all:
        Create missing tables on construction.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        echo: bool = False,
        create_all: bool = True,
        project_root: Path | None = None,
    ) -> None:
        self.url = url or default_database_url(project_root)
        connect_args = {}
        if self.url.startswith("sqlite"):
            # Allow use from the dashboard's worker threads.
            connect_args["check_same_thread"] = False

        self.engine: Engine = create_engine(
            self.url, echo=echo, future=True, connect_args=connect_args
        )
        if self.url.startswith("sqlite"):
            _enable_sqlite_pragmas(self.engine)

        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        if create_all:
            self.create_tables()
        log.debug("database ready", extra={"context": {"url": self._safe_url()}})

    def _safe_url(self) -> str:
        """The URL with any password redacted, safe to log."""
        if "@" in self.url and "//" in self.url:
            scheme, _, rest = self.url.partition("//")
            _, _, host = rest.partition("@")
            return f"{scheme}//***@{host}"
        return self.url

    def create_tables(self) -> None:
        """Create any missing tables."""
        Base.metadata.create_all(self.engine)

    def drop_tables(self) -> None:
        """Drop every Atlas table. Destructive; used by tests."""
        Base.metadata.drop_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Transactional session scope: commits on success, rolls back on error."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        """Close all pooled connections."""
        self.engine.dispose()

    def __enter__(self) -> DatabaseManager:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.dispose()


def _enable_sqlite_pragmas(engine: Engine) -> None:
    """Enable foreign keys and WAL mode on SQLite connections.

    SQLite does not enforce foreign keys unless asked, and the default rollback
    journal blocks readers while the dashboard is polling.
    """

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, connection_record) -> None:  # noqa: ARG001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()
