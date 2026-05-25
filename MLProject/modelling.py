import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils import estimator_html_repr


def parse_args():
    parser = argparse.ArgumentParser(description="Train customer churn model for MLflow CI.")
    parser.add_argument("--data-dir", default="customer_churn_preprocessing")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--subsample", type=float, default=1.0)
    return parser.parse_args()


def load_dataset(data_dir: Path):
    X_train = pd.read_csv(data_dir / "X_train.csv").astype(float)
    X_val = pd.read_csv(data_dir / "X_val.csv").astype(float)
    X_test = pd.read_csv(data_dir / "X_test.csv").astype(float)
    y_train = pd.read_csv(data_dir / "y_train.csv")["churn"]
    y_val = pd.read_csv(data_dir / "y_val.csv")["churn"]
    y_test = pd.read_csv(data_dir / "y_test.csv")["churn"]
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(model, X, y, prefix: str) -> dict[str, float]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        f"{prefix}_accuracy": accuracy_score(y, y_pred),
        f"{prefix}_precision": precision_score(y, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y, y_pred, zero_division=0),
        f"{prefix}_f1_score": f1_score(y, y_pred, zero_division=0),
        f"{prefix}_roc_auc": roc_auc_score(y, y_proba),
        f"{prefix}_log_loss": log_loss(y, y_proba),
    }


def save_confusion_matrix(y_true, y_pred, output_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["Not Churn", "Churn"])
    ax.set_yticks([0, 1], labels=["Not Churn", "Churn"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_roc_curve(y_true, y_proba, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("Validation ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_artifacts(model, X_train, X_val, y_val, X_test, y_test, metrics, output_dir: Path):
    artifacts_dir = output_dir / "artifacts"
    model_dir = output_dir / "model"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if model_dir.exists():
        shutil.rmtree(model_dir)

    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(model_dir),
        signature=signature,
        input_example=X_train.head(5),
    )

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    val_proba = model.predict_proba(X_val)[:, 1]

    save_confusion_matrix(
        y_val,
        val_pred,
        artifacts_dir / "validation_confusion_matrix.png",
        "Validation Confusion Matrix",
    )
    save_roc_curve(y_val, val_proba, artifacts_dir / "validation_roc_curve.png")

    feature_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(artifacts_dir / "feature_importance.csv", index=False)

    (artifacts_dir / "estimator.html").write_text(
        estimator_html_repr(model),
        encoding="utf-8",
    )
    (artifacts_dir / "metric_info.json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "classification_report_validation": classification_report(
                    y_val, val_pred, output_dict=True, zero_division=0
                ),
                "classification_report_test": classification_report(
                    y_test, test_pred, output_dict=True, zero_division=0
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return model_dir, artifacts_dir


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
    }

    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(data_dir)
    model = GradientBoostingClassifier(random_state=42, **params)

    with mlflow.start_run(run_name="customer_churn_ci_retraining"):
        mlflow.set_tags(
            {
                "dataset": "customer_churn_preprocessing",
                "workflow": "github_actions",
                "model_family": "GradientBoostingClassifier",
            }
        )
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        metrics = {}
        metrics.update(evaluate(model, X_train, y_train, "train"))
        metrics.update(evaluate(model, X_val, y_val, "validation"))
        metrics.update(evaluate(model, X_test, y_test, "test"))
        mlflow.log_metrics(metrics)

        model_dir, artifacts_dir = save_artifacts(
            model, X_train, X_val, y_val, X_test, y_test, metrics, output_dir
        )
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.head(5),
        )
        mlflow.log_artifacts(str(artifacts_dir), artifact_path="training_artifacts")

        run = mlflow.active_run()
        run_id = run.info.run_id
        summary = {
            "run_id": run_id,
            "model_uri": f"runs:/{run_id}/model",
            "local_model_dir": str(model_dir),
            "metrics": metrics,
            "params": params,
        }
        (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
