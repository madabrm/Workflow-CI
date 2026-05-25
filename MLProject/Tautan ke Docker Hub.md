# Tautan ke Docker Hub

Docker image akan dibuat otomatis oleh GitHub Actions menggunakan:

```bash
mlflow models build-docker -m outputs/model -n <dockerhub-username>/customer-churn-mlflow:latest
```

Setelah workflow berhasil, image tersedia di:

```text
https://hub.docker.com/r/madabrm/customer-churn-mlflow
```

Sebelum menjalankan workflow, tambahkan GitHub Secrets berikut:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
