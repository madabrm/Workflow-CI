# Workflow CI - Customer Churn MLflow Project

Folder ini berisi MLflow Project untuk retraining model customer churn secara otomatis melalui GitHub Actions.

## Struktur

```text
Workflow-CI
├── .github/workflows/mlflow-ci.yml
├── .workflow/mlflow-ci.yml
└── MLProject
    ├── MLProject
    ├── conda.yaml
    ├── modelling.py
    ├── customer_churn_preprocessing
    └── Tautan ke Docker Hub.md
```

## Menjalankan Lokal

```bash
cd Workflow-CI/MLProject
mlflow run . --env-manager local
```

## Menjalankan CI

Workflow berjalan pada `push`, `pull_request`, dan `workflow_dispatch`. Workflow akan:

1. Menginstal dependency.
2. Menjalankan retraining dengan `mlflow run`.
3. Mengunggah artifact model/training ke GitHub Actions artifact.
4. Membuat Docker image dari model MLflow dengan `mlflow models build-docker`.
5. Push image ke Docker Hub jika secret Docker Hub tersedia.
