"""
Model Training & Experiment Tracking Pipeline with MLflow.
Trains Random Forest, XGBoost, and LightGBM with class imbalance handling,
tracks experiments, and registers the production model pipeline.
"""

import sys
from pathlib import Path

# Force UTF-8 on Windows stdout/stderr to prevent emoji charmap encoding crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import joblib
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    recall_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn

from src.etl.db import PROJECT_ROOT
from src.ml.features import (
    load_dataset_from_warehouse,
    build_preprocessor_pipeline,
    get_chronological_splits,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("model_training")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "traffic-accident-severity-prediction")
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL", "accident-severity-classifier")
CHAMPION_ALIAS = "champion"
MLFLOW_LOCAL_DIR = Path(os.getenv("MLFLOW_LOCAL_DIR", str(PROJECT_ROOT)))
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models")))
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def plot_and_save_confusion_matrix(cm, classes, output_path: str):
    """Generates and saves confusion matrix heatmap plot artifact."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, yticklabels=classes,
        title='Confusion Matrix',
        ylabel='True Severity',
        xlabel='Predicted Severity'
    )
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def setup_mlflow():
    """
    Initializes MLflow tracking. Uses the MLflow server when it is reachable, otherwise a
    local SQLite backend (mlflow.db) with artifacts in mlruns/. A database backend is
    required for the Model Registry.
    """
    import socket

    server_online = False
    if MLFLOW_URI.startswith("http"):
        try:
            parts = MLFLOW_URI.replace("http://", "").replace("https://", "").split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 5000
            with socket.create_connection((host, port), timeout=0.5):
                server_online = True
        except Exception:
            server_online = False
            
    if server_online:
        mlflow.set_tracking_uri(MLFLOW_URI)
        logger.info(f"Connected to MLflow Tracking Server at: {MLFLOW_URI}")
    else:
        db_path = (MLFLOW_LOCAL_DIR / "mlflow.db").resolve().as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        logger.info(f"MLflow server offline. Tracking locally in {db_path} (artifacts in mlruns/).")
        if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
            mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=(MLFLOW_LOCAL_DIR / "mlruns").resolve().as_uri())

    mlflow.set_experiment(EXPERIMENT_NAME)


def log_and_register_model(pipeline):
    """Logs the inference pipeline, registers a new version and points the champion alias at it."""
    from mlflow.tracking import MlflowClient
    try:
        mlflow.sklearn.log_model(pipeline, name="model", registered_model_name=REGISTERED_MODEL_NAME,
                                 serialization_format="cloudpickle")
    except TypeError:
        # MLflow 2.x uses artifact_path instead of name
        mlflow.sklearn.log_model(pipeline, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAME,
                                 serialization_format="cloudpickle")
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    version = max(int(v.version) for v in versions)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, str(version))
    logger.info(f"Registered {REGISTERED_MODEL_NAME} version {version} with alias '{CHAMPION_ALIAS}'.")
    return version


def false_negative_rate(y_true, y_pred, cls: int) -> float:
    """FN / (FN + TP) for one class; 0.0 when the class is absent."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    positives = np.sum(y_true == cls)
    if positives == 0:
        return 0.0
    return float(np.sum((y_true == cls) & (y_pred != cls)) / positives)


def train_and_evaluate_models():
    """
    Main training workflow:
    1. Loads data from warehouse.
    2. Chronological Train/Val/Test split.
    3. Fits preprocessor.
    4. Trains Random Forest, XGBoost, and LightGBM with class weights.
    5. Logs metrics, params, and artifacts to MLflow.
    6. Selects and serializes best production candidate.
    """
    setup_mlflow()
    
    df = load_dataset_from_warehouse()
    if len(df) < 50:
        logger.error("Insufficient records in warehouse to train model. Run ETL first.")
        return
        
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_chronological_splits(df)
    
    # Import SMOTE and accuracy_score
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import accuracy_score
    
    # Preprocessor pipeline
    preprocessor = build_preprocessor_pipeline()
    logger.info("Fitting preprocessor on chronological training split...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    # 1. Apply SMOTE to synthesize minority samples for Fatal and Serious classes
    k_neighbors = min(4, sum(y_train == 2) - 1)
    if k_neighbors >= 1:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)
        logger.info(f"Applied SMOTE: Resampled training set from {len(y_train)} to {len(y_train_res)} records.")
    else:
        X_train_res, y_train_res = X_train_proc, y_train
        logger.info("SMOTE skipped due to insufficient minority class samples.")
        
    # Compute balanced class weights
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {cls: w for cls, w in zip(classes, weights)}
    logger.info(f"Calculated class weights: {class_weight_dict}")
    
    # Candidate models (fixed hyperparameters)
    models = {
        # Tree size is capped so the serialized model stays small on ~800k SMOTE rows
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=20,
            max_samples=0.3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=180,
            max_depth=7,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.85,
            class_weight="balanced",
            random_state=42,
            verbosity=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=160,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            num_class=3,
            random_state=42,
            eval_metric="mlogloss"
        )
    }
    
    class_names = ["Slight", "Serious", "Fatal"]
    best_model_name = None
    best_macro_f1 = -1.0
    best_pipeline = None
    
    for name, model in models.items():
        logger.info(f"--- Training {name} (SMOTE-resampled training split) ---")
        with mlflow.start_run(run_name=name):
            # Fit model on SMOTE resampled feature space
            model.fit(X_train_res, y_train_res)
                
            # Predictions on Validation
            y_pred_val = model.predict(X_val_proc)
            y_proba_val = model.predict_proba(X_val_proc)
            
            # Compute comprehensive metrics
            val_accuracy = accuracy_score(y_val, y_pred_val)
            macro_f1 = f1_score(y_val, y_pred_val, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_val, y_pred_val, average="weighted", zero_division=0)
            fatal_recall = recall_score(y_val, y_pred_val, labels=[2], average="macro", zero_division=0)
            serious_recall = recall_score(y_val, y_pred_val, labels=[1], average="macro", zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_val, y_proba_val, multi_class="ovr", average="macro")
            except Exception:
                roc_auc = 0.5
                
            # Log params & metrics prominently in MLflow
            mlflow.log_params({
                "model_name": name,
                "sampling_strategy": "SMOTE",
                "train_samples_raw": len(X_train),
                "train_samples_resampled": len(X_train_res),
                "features_count": X_train_proc.shape[1]
            })
            
            mlflow.log_metrics({
                "overall_accuracy": val_accuracy,
                "val_macro_f1": macro_f1,
                "val_weighted_f1": weighted_f1,
                "val_fatal_recall": fatal_recall,
                "val_serious_recall": serious_recall,
                "val_roc_auc": roc_auc
            })
            
            logger.info(f"{name} Results - Overall Accuracy: {val_accuracy:.4f}, Macro F1: {macro_f1:.4f}, Fatal Recall: {fatal_recall:.4f}, ROC-AUC: {roc_auc:.4f}")
            
            # Confusion Matrix Artifact
            cm = confusion_matrix(y_val, y_pred_val, labels=[0, 1, 2])
            cm_path = str(MODEL_DIR / f"cm_{name.lower()}.png")
            plot_and_save_confusion_matrix(cm, class_names, cm_path)
            mlflow.log_artifact(cm_path)
            
            # Select best candidate
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_model_name = name
                best_pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("classifier", model)
                ])
                
    logger.info(f"=== Best Model Selected: {best_model_name} with Val Macro-F1: {best_macro_f1:.4f} ===")
    
    # Final evaluation on hold-out test set
    y_pred_test = best_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_macro_f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
    test_fatal_fnr = false_negative_rate(y_test, y_pred_test, 2)
    test_serious_fnr = false_negative_rate(y_test, y_pred_test, 1)
    try:
        test_roc_auc = float(roc_auc_score(y_test, best_pipeline.predict_proba(X_test), multi_class="ovr", average="macro"))
    except Exception:
        test_roc_auc = None
    test_report = classification_report(y_test, y_pred_test, labels=[0, 1, 2], target_names=class_names,
                                        output_dict=True, zero_division=0)
    test_report["overall_accuracy"] = test_accuracy
    test_report["champion_model"] = best_model_name
    test_report["fatal_fnr"] = test_fatal_fnr
    test_report["serious_fnr"] = test_serious_fnr
    test_report["roc_auc_ovr"] = test_roc_auc
    test_report["split_sizes"] = {"train": len(X_train), "val": len(X_val), "test": len(X_test)}

    test_cm_path = str(MODEL_DIR / "cm_test_champion.png")
    plot_and_save_confusion_matrix(confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2]), class_names, test_cm_path)

    # Log champion model to MLflow and register it
    with mlflow.start_run(run_name=f"Production_{best_model_name}") as run:
        mlflow.log_metrics({
            "overall_accuracy": test_accuracy,
            "test_macro_f1": test_macro_f1,
            "test_weighted_f1": test_weighted_f1,
            "test_fatal_recall": 1 - test_fatal_fnr,
            "test_serious_recall": 1 - test_serious_fnr,
            "test_fatal_fnr": test_fatal_fnr,
            "test_serious_fnr": test_serious_fnr,
        })
        if test_roc_auc is not None:
            mlflow.log_metric("test_roc_auc", test_roc_auc)
        mlflow.log_param("champion_model", best_model_name)
        mlflow.set_tag("imbalance_technique", "SMOTE + class weights")
        mlflow.log_artifact(test_cm_path)
        registered_version = log_and_register_model(best_pipeline)
        run_id = run.info.run_id

    model_save_path = MODEL_DIR / "accident_severity_model.joblib"
    joblib.dump(best_pipeline, model_save_path, compress=3)
    logger.info(f"Saved complete inference pipeline to: {model_save_path}")

    import json
    import datetime
    report_path = MODEL_DIR / "test_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)
    logger.info(f"Saved test evaluation report to: {report_path}")

    metadata = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "registered_version": registered_version,
        "alias": CHAMPION_ALIAS,
        "model_version": f"{REGISTERED_MODEL_NAME}/v{registered_version}",
        "mlflow_run_id": run_id,
        "champion_algorithm": best_model_name,
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "training_rows": len(X_train),
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "baseline_test_metrics": {
            "macro_f1": test_macro_f1,
            "accuracy": test_accuracy,
            "fatal_fnr": test_fatal_fnr,
            "serious_fnr": test_serious_fnr,
        },
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Test results - Macro F1: {test_macro_f1:.4f}, Fatal recall: {1 - test_fatal_fnr:.4f}, "
                f"Serious recall: {1 - test_serious_fnr:.4f}")
    return best_model_name, test_macro_f1


if __name__ == "__main__":
    # Non-zero exit lets the Airflow retrain task fail visibly when there is nothing to train on
    sys.exit(0 if train_and_evaluate_models() is not None else 1)
