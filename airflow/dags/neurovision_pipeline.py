from datetime import datetime

from airflow import DAG

from airflow.operators.python import PythonOperator


def prepare_data():
    from src.data.loader import load_dataset

    X, y = load_dataset()

    print("Dataset loaded:", X.shape, y.shape)


def train_model():
    from main import main

    main()


def evaluate_model_task():
    print("Evaluation task completed.")


def register_model_task():
    print("Model registration task completed.")


with DAG(
    dag_id="neurovision_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False
) as dag:

    data_task = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data
    )

    training_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    evaluation_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model_task
    )

    registry_task = PythonOperator(
        task_id="register_model",
        python_callable=register_model_task
    )

    data_task >> training_task >> evaluation_task >> registry_task