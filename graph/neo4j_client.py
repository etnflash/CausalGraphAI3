"""Neo4j client — thin wrapper around the official Neo4j Python driver."""

import logging
from contextlib import contextmanager
from typing import Any, Generator

from app.config import settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Manages the connection pool to a Neo4j instance.

    Usage::

        client = Neo4jClient()
        with client.session() as session:
            session.run("MATCH (n) RETURN count(n)")
        client.close()
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        self._uri = uri or settings.neo4j_uri
        self._user = user or settings.neo4j_user
        self._password = password or settings.neo4j_password
        self._driver = None

    def _get_driver(self):
        """Lazily initialise the Neo4j driver."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase  # noqa: PLC0415

                self._driver = GraphDatabase.driver(
                    self._uri,
                    auth=(self._user, self._password),
                )
                logger.info("Connected to Neo4j at %s", self._uri)
            except ImportError as exc:
                raise ImportError(
                    "neo4j is required. Install it with: pip install neo4j"
                ) from exc
        return self._driver

    @contextmanager
    def session(self) -> Generator:
        """Yield a Neo4j session, closing it on exit."""
        driver = self._get_driver()
        neo4j_session = driver.session()
        try:
            yield neo4j_session
        finally:
            neo4j_session.close()

    def run_query(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict]:
        """Execute a Cypher query and return all records as dicts.

        Args:
            cypher: Cypher query string.
            parameters: Optional query parameters.

        Returns:
            List of result records as plain dicts.
        """
        with self.session() as s:
            result = s.run(cypher, parameters or {})
            return [record.data() for record in result]

    def verify_connectivity(self) -> bool:
        """Return ``True`` if the database is reachable, ``False`` otherwise."""
        try:
            self._get_driver().verify_connectivity()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j connectivity check failed: %s", exc)
            return False

    def close(self) -> None:
        """Close the underlying driver and release all connections."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed.")
