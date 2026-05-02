import asyncio
import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from dotenv import load_dotenv

load_dotenv()

# Symbols to track
SYMBOLS = [
    "RELIANCE.BSE",
    "TCS.BSE",
    "INFY.BSE",
]
TABLE_NAME = "stock_prices"

logger = logging.getLogger(__name__)

default_args = {
    "owner":            "saketh",
    "depends_on_past":  False,
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

dag = DAG(
    dag_id="stock_etl_nightly",
    description="Pulls daily stock data from Alpha Vantage and loads into Supabase",
    schedule_interval="0 3 * * 1-5",   # 3:00 AM, Monday–Friday only
    start_date=days_ago(1),
    catchup=False,
    default_args=default_args,
    tags=["etl", "stocks"],
)


def run_pipeline_for_symbol(symbol: str, **context):
    """Task callable — runs the async pipeline synchronously for Airflow."""
    from ETL.pipeline import ETLPipeline

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    db_url  = os.getenv("SUPABASE_URL")
    db_key  = os.getenv("SUPABASE_KEY")

    if not all([api_key, db_url, db_key]):
        raise ValueError("Missing required environment variables.")

    pipeline = ETLPipeline(api_key, db_url, db_key)
    result = asyncio.run(pipeline.run(symbol, TABLE_NAME))

    logger.info(f"[{symbol}] Result: {result}")

    # Push result to XCom so you can inspect it in Airflow UI
    context["ti"].xcom_push(key="result", value=result)

    if result["status"] == "skipped":
        logger.info(f"[{symbol}] Skipped: {result['reason']}")
    elif result["status"] == "inserted":
        logger.info(f"[{symbol}] Done: {result['rows']} rows inserted.")


# Dynamically create one task per symbol
for symbol in SYMBOLS:
    task_id = f"etl_{symbol.replace('.', '_').lower()}"

    PythonOperator(
        task_id=task_id,
        python_callable=run_pipeline_for_symbol,
        op_kwargs={"symbol": symbol},
        dag=dag,
    )