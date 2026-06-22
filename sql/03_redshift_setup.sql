-- Amazon Redshift data warehouse setup
-- Optimized with distribution and sort keys for query performance

-- Raw churn data table
CREATE TABLE IF NOT EXISTS public.customer_churn (
    customer_id        VARCHAR(20)   NOT NULL ENCODE zstd,
    gender             VARCHAR(10)   ENCODE bytedict,
    senior_citizen     SMALLINT      DEFAULT 0,
    partner            SMALLINT      DEFAULT 0,
    dependents         SMALLINT      DEFAULT 0,
    tenure             INTEGER,
    phone_service      SMALLINT      DEFAULT 1,
    multiple_lines     VARCHAR(25)   ENCODE bytedict,
    internet_service   VARCHAR(20)   ENCODE bytedict,
    online_security    VARCHAR(25)   ENCODE bytedict,
    online_backup      VARCHAR(25)   ENCODE bytedict,
    device_protection  VARCHAR(25)   ENCODE bytedict,
    tech_support       VARCHAR(25)   ENCODE bytedict,
    streaming_tv       VARCHAR(25)   ENCODE bytedict,
    streaming_movies   VARCHAR(25)   ENCODE bytedict,
    contract           VARCHAR(30)   ENCODE bytedict,
    paperless_billing  SMALLINT      DEFAULT 0,
    payment_method     VARCHAR(40)   ENCODE bytedict,
    monthly_charges    DECIMAL(8,2),
    total_charges      DECIMAL(10,2),
    churn              SMALLINT      DEFAULT 0,
    signup_date        TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (customer_id)
)
DISTSTYLE KEY
DISTKEY (customer_id)
SORTKEY (signup_date, contract);

-- Prediction results table
CREATE TABLE IF NOT EXISTS public.churn_predictions (
    customer_id        VARCHAR(20)   NOT NULL,
    churn_probability  DECIMAL(5,4),
    churn_prediction   SMALLINT,
    risk_level         VARCHAR(10)   ENCODE bytedict,
    model_version      VARCHAR(20)   DEFAULT '1.0.0',
    predicted_at       TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
)
DISTSTYLE KEY
DISTKEY (customer_id)
SORTKEY (predicted_at);

-- COPY from S3 example
-- COPY public.customer_churn
-- FROM 's3://your-bucket/data/telco_churn.csv'
-- IAM_ROLE 'arn:aws:iam::123456789012:role/RedshiftS3Role'
-- CSV
-- IGNOREHEADER 1
-- TIMEFORMAT 'auto';

-- Feature importance aggregation query
CREATE OR REPLACE VIEW public.v_churn_feature_impact AS
SELECT
    contract,
    internet_service,
    payment_method,
    COUNT(*) AS customer_count,
    ROUND(AVG(CAST(churn AS DECIMAL)), 4) AS churn_rate,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly,
    ROUND(AVG(tenure), 1) AS avg_tenure
FROM public.customer_churn
GROUP BY 1, 2, 3
ORDER BY churn_rate DESC;
