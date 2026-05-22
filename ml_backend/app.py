from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ml_pipeline import (
    CLF_PARAMS,
    CLUS_PARAMS,
    REG_PARAMS,
    analyze_data,
    evaluate_model,
    get_feature_importance,
    run_preprocessing,
    save_model,
    train_models,
)


BASE_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = BASE_DIR / "sessions"
MODELS_DIR = BASE_DIR / "saved_models"
SESSIONS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

ALGORITHM_OPTIONS = {
    "classification": ["logistic_regression", "random_forest", "svm", "xgboost"],
    "regression": ["linear_regression", "ridge", "lasso", "random_forest"],
    "clustering": ["kmeans", "dbscan"],
}


app = FastAPI(
    title="Automated ML API",
    description="Upload data, analyze issues, train models, and download the saved model bundle.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    session_id: str
    task_type: str = Field(pattern="^(classification|regression|clustering)$")
    target_col: str | None = None


class TrainRequest(BaseModel):
    session_id: str
    task_type: str = Field(pattern="^(classification|regression|clustering)$")
    target_col: str | None = None
    choices: dict[str, Any] = Field(default_factory=dict)
    selected_algorithms: list[str] | None = None
    use_default_algorithms: bool = True
    use_grid_search: bool = True
    custom_params: dict[str, dict[str, list[Any]]] | None = None


def _session_file(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def _load_session(session_id: str) -> dict[str, Any]:
    session_path = _session_file(session_id)
    if not session_path.exists():
        raise HTTPException(status_code=404, detail="Session not found.")
    return json.loads(session_path.read_text(encoding="utf-8"))


def _save_session(session_id: str, payload: dict[str, Any]) -> None:
    _session_file(session_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_dataframe(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    raise HTTPException(status_code=400, detail="Only .csv, .xlsx, and .xls files are supported.")


def _serialize_issues(issues: dict[str, Any]) -> list[dict[str, Any]]:
    cards = []
    for issue_key, payload in issues.items():
        cards.append(
            {
                "id": issue_key,
                "question": payload.get("question"),
                "detail": payload.get("detail"),
                "options": payload.get("options", {}),
                "defaults": payload.get("defaults", {}),
                "actionable": True,
            }
        )
    return cards


def _default_choices(task_type: str) -> dict[str, Any]:
    choices = {
        "duplicates": True,
        "missing_values": True,
        "strategy_numeric": "median",
        "strategy_categorical": "most_frequent",
        "outliers": task_type in {"classification", "regression"},
        "outlier_method": "iqr_cap",
        "outlier_iqr_k": 1.5,
        "categorical_encoding": True,
        "encoding_strategy": "onehot",
        "high_cardinality": True,
        "high_cardinality_strategy": "drop",
        "scaling": True,
        "scaler": "standard",
        "class_imbalance": False,
        "resample_strategy": "smote",
        "high_dimensionality": False,
        "pca_n_components": None,
        "random_state": 42,
    }
    if task_type in {"classification", "regression"}:
        choices["test_size"] = 0.2
    return choices


def _default_param_space(task_type: str) -> dict[str, Any]:
    if task_type == "classification":
        return CLF_PARAMS
    if task_type == "regression":
        return REG_PARAMS
    return CLUS_PARAMS


def _validate_target(df: pd.DataFrame, task_type: str, target_col: str | None) -> None:
    if task_type in {"classification", "regression"}:
        if not target_col:
            raise HTTPException(status_code=400, detail="target_col is required for classification and regression.")
        if target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found.")


def _resolve_algorithms(task_type: str, selected_algorithms: list[str] | None, use_default_algorithms: bool) -> list[str]:
    available = ALGORITHM_OPTIONS[task_type]
    if use_default_algorithms or not selected_algorithms:
        return available

    invalid = [algo for algo in selected_algorithms if algo not in available]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unsupported algorithms for {task_type}: {invalid}")

    if len(selected_algorithms) < 2:
        raise HTTPException(status_code=400, detail="Please choose at least two algorithms for this task.")

    return selected_algorithms


def _model_filename(session_id: str) -> str:
    return f"{session_id}_model.joblib"


@app.get("/")
def healthcheck() -> dict[str, str]:
    return {"message": "Automated ML API is running."}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Only .csv, .xlsx, and .xls files are supported.")

    session_id = str(uuid.uuid4())
    saved_path = SESSIONS_DIR / f"{session_id}{suffix}"
    saved_path.write_bytes(await file.read())

    df = _read_dataframe(saved_path)
    session_payload = {
        "session_id": session_id,
        "file_path": str(saved_path),
        "original_filename": file.filename,
    }
    _save_session(session_id, session_payload)

    return {
        "session_id": session_id,
        "filename": file.filename,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "preview": df.head(5).replace({pd.NA: None}).where(pd.notnull(df.head(5)), None).to_dict(orient="records"),
        "message": "File uploaded successfully. You can now ask the backend to analyze the dataset.",
    }


@app.post("/analyze")
def analyze_dataset(request: AnalyzeRequest) -> dict[str, Any]:
    session = _load_session(request.session_id)
    df = _read_dataframe(Path(session["file_path"]))
    _validate_target(df, request.task_type, request.target_col)

    issues = analyze_data(df, request.task_type, request.target_col)
    cards = _serialize_issues(issues)

    model_questions = [
        {
            "id": "build_model",
            "question": "Issues detected. Should I solve them and then build the model?",
            "detail": "If yes, the backend will apply the choices you send in /train.",
            "actionable": True,
        },
        {
            "id": "algorithm_selection",
            "question": "Do you want to choose specific algorithms or use the app defaults?",
            "detail": {
                "available_algorithms": ALGORITHM_OPTIONS[request.task_type],
                "default_algorithms": ALGORITHM_OPTIONS[request.task_type],
            },
            "actionable": True,
        },
        {
            "id": "search_space_selection",
            "question": "Do you want to use default search spaces or send your own?",
            "detail": _default_param_space(request.task_type),
            "actionable": True,
        },
    ]

    if request.task_type in {"classification", "regression"}:
        model_questions.append(
            {
                "id": "test_split_confirmation",
                "question": "Should I split the data into train and test sets before training?",
                "detail": {"default_test_size": 0.2},
                "actionable": True,
            }
        )

    session.update(
        {
            "task_type": request.task_type,
            "target_col": request.target_col,
            "analysis": issues,
        }
    )
    _save_session(request.session_id, session)

    return {
        "session_id": request.session_id,
        "task_type": request.task_type,
        "target_col": request.target_col,
        "questions_for_user": cards + model_questions,
        "available_algorithms": ALGORITHM_OPTIONS[request.task_type],
        "default_choices": _default_choices(request.task_type),
        "default_param_space": _default_param_space(request.task_type),
        "message": "Analysis complete. Show these questions to the user as actionable recommendation cards.",
    }


@app.post("/train")
def train_endpoint(request: TrainRequest) -> dict[str, Any]:
    session = _load_session(request.session_id)
    df = _read_dataframe(Path(session["file_path"]))
    _validate_target(df, request.task_type, request.target_col)

    final_choices = _default_choices(request.task_type)
    final_choices.update(request.choices)

    selected_algorithms = _resolve_algorithms(
        request.task_type,
        request.selected_algorithms,
        request.use_default_algorithms,
    )

    try:
        X_train, X_test, y_train, y_test, feature_names, artifacts, preprocess_report = run_preprocessing(
            df=df,
            task_type=request.task_type,
            target_col=request.target_col,
            choices=final_choices,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {exc}") from exc

    results, best_model, best_name, best_score = train_models(
        X_train=X_train,
        y_train=y_train,
        task_type=request.task_type,
        selected_algos=selected_algorithms,
        use_grid_search=request.use_grid_search,
        custom_params=request.custom_params,
        random_state=final_choices.get("random_state", 42),
    )

    if best_model is None:
        raise HTTPException(status_code=400, detail="No model could be trained with the provided configuration.")

    evaluation_input = X_test if request.task_type != "clustering" else X_train
    evaluation = evaluate_model(
        best_model,
        evaluation_input,
        y_test,
        request.task_type,
        feature_names=feature_names,
    )

    model_path = MODELS_DIR / _model_filename(request.session_id)
    save_model(
        model=best_model,
        feature_names=feature_names,
        task_type=request.task_type,
        target_col=request.target_col,
        artifacts=artifacts,
        path=model_path,
    )

    comparison = {}
    for name, payload in results.items():
        comparison[name] = {
            "cv_score": payload.get("cv_score"),
            "best_params": payload.get("best_params", {}),
            "error": payload.get("error"),
        }

    session.update(
        {
            "task_type": request.task_type,
            "target_col": request.target_col,
            "model_path": str(model_path),
            "selected_algorithms": selected_algorithms,
            "choices": final_choices,
        }
    )
    _save_session(request.session_id, session)

    return {
        "session_id": request.session_id,
        "preprocessing_report": preprocess_report,
        "selected_algorithms": selected_algorithms,
        "model_comparison": comparison,
        "best_model": {
            "name": best_name,
            "score": best_score,
        },
        "evaluation": evaluation,
        "feature_importance": get_feature_importance(best_model, feature_names),
        "download_url": f"/download/{request.session_id}",
        "message": "Training complete. The model bundle is ready to download.",
    }


@app.get("/download/{session_id}")
def download_model(session_id: str) -> FileResponse:
    session = _load_session(session_id)
    model_path = Path(session.get("model_path", ""))
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Saved model not found for this session.")

    return FileResponse(
        path=model_path,
        media_type="application/octet-stream",
        filename=model_path.name,
    )
