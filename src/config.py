import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    # App
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_PORT: int = int(os.getenv("APP_PORT", 8000))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # PostgreSQL
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", 5432))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "churn_db")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"postgresql://postgres:@localhost:5432/churn_db"
    )

    # Snowflake
    SNOWFLAKE_ACCOUNT: str = os.getenv("SNOWFLAKE_ACCOUNT", "")
    SNOWFLAKE_USER: str = os.getenv("SNOWFLAKE_USER", "")
    SNOWFLAKE_PASSWORD: str = os.getenv("SNOWFLAKE_PASSWORD", "")
    SNOWFLAKE_WAREHOUSE: str = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
    SNOWFLAKE_DATABASE: str = os.getenv("SNOWFLAKE_DATABASE", "CHURN_DB")
    SNOWFLAKE_SCHEMA: str = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
    SNOWFLAKE_ROLE: str = os.getenv("SNOWFLAKE_ROLE", "SYSADMIN")

    # AWS / Redshift
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    REDSHIFT_HOST: str = os.getenv("REDSHIFT_HOST", "")
    REDSHIFT_PORT: int = int(os.getenv("REDSHIFT_PORT", 5439))
    REDSHIFT_DB: str = os.getenv("REDSHIFT_DB", "churn_dw")
    REDSHIFT_USER: str = os.getenv("REDSHIFT_USER", "")
    REDSHIFT_PASSWORD: str = os.getenv("REDSHIFT_PASSWORD", "")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "")
    S3_MODEL_PREFIX: str = os.getenv("S3_MODEL_PREFIX", "models/")

    # Model
    MODEL_ARTIFACT_PATH: Path = BASE_DIR / os.getenv("MODEL_ARTIFACT_PATH", "models/artifacts")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHURN_MODEL_FILE: str = os.getenv("CHURN_MODEL_FILE", "churn_model_pipeline.joblib")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", 384))

    # Data paths
    RAW_DATA_PATH: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_PATH: Path = BASE_DIR / "data" / "processed"

    def __init__(self):
        self.MODEL_ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
        self.RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)


settings = Settings()
