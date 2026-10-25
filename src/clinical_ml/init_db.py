from pathlib import Path

from .prediction_repository import get_connection


SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "sql"
    / "schema.sql"
)


def initialize_database() -> None:
    """Create the database tables required by the application."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    with get_connection() as connection:
        connection.execute(schema_sql)

    print("Database schema initialized successfully.")


if __name__ == "__main__":
    initialize_database()