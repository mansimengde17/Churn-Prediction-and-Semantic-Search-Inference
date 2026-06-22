"""
AWS Redshift connector.
Used for pulling large-scale historical training data from the data warehouse.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from src.config import settings


class RedshiftConnector:
    """
    Connects to Amazon Redshift using redshift-connector.
    Can also use SQLAlchemy + psycopg2 for ORM access.
    """

    def __init__(self, **overrides):
        self.host = overrides.get("host", settings.REDSHIFT_HOST)
        self.port = overrides.get("port", settings.REDSHIFT_PORT)
        self.database = overrides.get("database", settings.REDSHIFT_DB)
        self.user = overrides.get("user", settings.REDSHIFT_USER)
        self.password = overrides.get("password", settings.REDSHIFT_PASSWORD)
        self._conn = None

    def connect(self):
        try:
            import redshift_connector

            self._conn = redshift_connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port,
            )
            logger.info(f"Connected to Redshift: {self.host}/{self.database}")
        except ImportError:
            raise RuntimeError(
                "redshift-connector not installed. "
                "Run: pip install redshift-connector"
            )

    def run_query(self, query: str) -> pd.DataFrame:
        if self._conn is None:
            self.connect()
        cursor = self._conn.cursor()
        cursor.execute(query)
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        df = pd.DataFrame(rows, columns=cols)
        logger.info(f"Redshift query returned {len(df):,} rows")
        return df

    def fetch_training_data(self, schema: str = "public", table: str = "customer_churn") -> pd.DataFrame:
        return self.run_query(f"SELECT * FROM {schema}.{table}")

    def copy_from_s3(self, table: str, s3_path: str, iam_role: str, file_format: str = "CSV"):
        """COPY command to bulk-load data from S3 to Redshift."""
        sql = f"""
            COPY {table}
            FROM '{s3_path}'
            IAM_ROLE '{iam_role}'
            FORMAT AS {file_format}
            IGNOREHEADER 1
            TIMEFORMAT 'auto'
        """
        if self._conn is None:
            self.connect()
        cursor = self._conn.cursor()
        cursor.execute(sql)
        self._conn.commit()
        logger.info(f"COPY from {s3_path} to {table} complete")

    def unload_to_s3(self, query: str, s3_path: str, iam_role: str):
        """UNLOAD query results to S3 as Parquet."""
        sql = f"""
            UNLOAD ('{query}')
            TO '{s3_path}'
            IAM_ROLE '{iam_role}'
            FORMAT AS PARQUET
            ALLOWOVERWRITE
        """
        if self._conn is None:
            self.connect()
        cursor = self._conn.cursor()
        cursor.execute(sql)
        self._conn.commit()
        logger.info(f"UNLOAD complete → {s3_path}")

    def close(self):
        if self._conn:
            self._conn.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def example_queries() -> dict:
        return {
            "cohort_churn_analysis": """
                SELECT
                    DATE_TRUNC('quarter', signup_date) AS cohort,
                    DATEDIFF('month', signup_date, GETDATE()) AS months_since_signup,
                    COUNT(*) AS cohort_size,
                    SUM(churn::int) AS churned,
                    ROUND(100.0 * SUM(churn::int) / COUNT(*), 2) AS churn_rate
                FROM public.customer_churn
                GROUP BY 1, 2
                ORDER BY 1, 2
            """,
            "feature_distribution": """
                SELECT
                    contract_type,
                    internet_service,
                    AVG(monthly_charges) AS avg_monthly,
                    AVG(tenure) AS avg_tenure,
                    AVG(churn::float) AS churn_rate
                FROM public.customer_churn
                GROUP BY 1, 2
                ORDER BY churn_rate DESC
            """,
        }
