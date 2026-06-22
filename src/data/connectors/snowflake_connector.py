"""
Snowflake data warehouse connector.
Used for pulling training data from enterprise data warehouse.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from src.config import settings


class SnowflakeConnector:
    """
    Connects to Snowflake using the snowflake-connector-python library.
    Credentials are sourced from environment variables / .env file.
    """

    def __init__(self, **overrides):
        self.connection_params = {
            "account": overrides.get("account", settings.SNOWFLAKE_ACCOUNT),
            "user": overrides.get("user", settings.SNOWFLAKE_USER),
            "password": overrides.get("password", settings.SNOWFLAKE_PASSWORD),
            "warehouse": overrides.get("warehouse", settings.SNOWFLAKE_WAREHOUSE),
            "database": overrides.get("database", settings.SNOWFLAKE_DATABASE),
            "schema": overrides.get("schema", settings.SNOWFLAKE_SCHEMA),
            "role": overrides.get("role", settings.SNOWFLAKE_ROLE),
        }
        self._conn = None
        self._engine = None

    def connect(self):
        try:
            import snowflake.connector

            self._conn = snowflake.connector.connect(**self.connection_params)
            logger.info(
                f"Connected to Snowflake: {self.connection_params['account']} / "
                f"{self.connection_params['database']}.{self.connection_params['schema']}"
            )
        except ImportError:
            raise RuntimeError(
                "snowflake-connector-python not installed. "
                "Run: pip install snowflake-connector-python"
            )

    def get_sqlalchemy_engine(self):
        """Return a SQLAlchemy engine for use with pandas read_sql."""
        from snowflake.sqlalchemy import URL
        from sqlalchemy import create_engine

        url = URL(
            account=self.connection_params["account"],
            user=self.connection_params["user"],
            password=self.connection_params["password"],
            database=self.connection_params["database"],
            schema=self.connection_params["schema"],
            warehouse=self.connection_params["warehouse"],
            role=self.connection_params["role"],
        )
        self._engine = create_engine(url)
        return self._engine

    def fetch_training_data(self, table: str = "CUSTOMER_CHURN") -> pd.DataFrame:
        """Pull full training table from Snowflake."""
        query = f"SELECT * FROM {self.connection_params['schema']}.{table}"
        return self.run_query(query)

    def run_query(self, query: str) -> pd.DataFrame:
        if self._conn is None:
            self.connect()
        cur = self._conn.cursor()
        cur.execute(query)
        df = cur.fetch_pandas_all()
        logger.info(f"Snowflake query returned {len(df):,} rows")
        return df

    def upload_predictions(self, df: pd.DataFrame, table: str = "CHURN_PREDICTIONS"):
        """Write predictions back to Snowflake."""
        if self._engine is None:
            self.get_sqlalchemy_engine()
        df.to_sql(table.lower(), self._engine, if_exists="replace", index=False)
        logger.info(f"Uploaded {len(df):,} predictions to {table}")

    def close(self):
        if self._conn:
            self._conn.close()
        if self._engine:
            self._engine.dispose()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def example_queries() -> dict:
        return {
            "monthly_churn_rate": """
                SELECT
                    DATE_TRUNC('month', created_date) AS month,
                    COUNT(*) AS total_customers,
                    SUM(CASE WHEN churn = 1 THEN 1 ELSE 0 END) AS churned,
                    ROUND(100.0 * SUM(CASE WHEN churn = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
                FROM CUSTOMER_CHURN
                GROUP BY 1
                ORDER BY 1 DESC
            """,
            "high_risk_customers": """
                SELECT customer_id, monthly_charges, contract, internet_service
                FROM CHURN_PREDICTIONS
                WHERE churn_probability > 0.70
                ORDER BY churn_probability DESC
                LIMIT 500
            """,
            "revenue_at_risk": """
                SELECT
                    SUM(monthly_charges) AS revenue_at_risk,
                    COUNT(*) AS customers_at_risk
                FROM CHURN_PREDICTIONS
                WHERE churn_probability > 0.50
            """,
        }
