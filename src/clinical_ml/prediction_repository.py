import os

import psycopg


def get_connection() -> psycopg.Connection:
    """Create a connection to the clinical ML database."""
    password = os.getenv("DB_PASSWORD")

    if not password:
        raise RuntimeError(
            "DB_PASSWORD environment variable is not set."
        )

    return psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "clinical_ml"),
        user=os.getenv("DB_USER", "clinical_ml_app"),
        password=password,
    )


def save_prediction(
    *,
    model_version: str,
    features: list[float],
    diagnosis: str,
    confidence: float,
) -> int:
    """Store one prediction and return its database ID."""
    query = """
        INSERT INTO predictions (
            model_version,
            features,
            diagnosis,
            confidence
        )
        VALUES (%s, %s, %s, %s)
        RETURNING id
    """

    parameters = (
        model_version,
        features,
        diagnosis,
        confidence,
    )

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            row = cursor.fetchone()

    if row is None:
        raise RuntimeError(
            "PostgreSQL did not return a prediction ID."
        )

    return int(row[0])


def check_connection() -> None:
    """Read basic information from PostgreSQL."""
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    current_database(),
                    current_user,
                    (SELECT COUNT(*) FROM predictions)
                """
            )

            row = cursor.fetchone()

            if row is None:
                raise RuntimeError(
                    "PostgreSQL returned no connection information."
                )

            database_name, user_name, prediction_count = row

            print(f"Database: {database_name}")
            print(f"User: {user_name}")
            print(f"Prediction count: {prediction_count}")


if __name__ == "__main__":
    check_connection()