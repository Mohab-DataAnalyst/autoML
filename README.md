# AutoML Tool

AutoML Tool is an end-to-end AutoML-style web application for uploading tabular datasets, reviewing data quality recommendations, training multiple machine learning models, comparing results, and exporting the best trained model as a reusable `.joblib` bundle.

The project is split into two main parts:

- `data_eng_ml/`: the ML backend and FastAPI service.
- `frontend/`: a React + TypeScript + Material UI wizard for upload, task configuration, recommendation review, training, results, and model download.

## What I Worked On

I collaborated on the project by building the ML backend layer in `data_eng_ml/`, including the core machine learning pipeline and API integration. My work focused on turning uploaded datasets into an automated, user-guided modeling workflow:

- Built the automated data analysis pipeline that detects missing values, duplicate rows, categorical columns, high-cardinality features, outliers, class imbalance, and high-dimensional datasets.
- Implemented configurable preprocessing for supervised and unsupervised learning, including imputation, duplicate removal, categorical encoding, high-cardinality handling, IQR-based outlier capping, feature scaling, class resampling, PCA, and train/test splitting.
- Added model training workflows for classification, regression, and clustering using scikit-learn, imbalanced-learn, and XGBoost.
- Implemented default and custom hyperparameter search spaces with GridSearchCV for supervised models and silhouette-based selection for clustering.
- Added evaluation outputs for classification, regression, and clustering, including confusion matrices, regression metrics, clustering quality scores, cluster labels/profiles, and feature importance where supported.
- Exposed the ML workflow through FastAPI endpoints for dataset upload, analysis, training, and model download.

## Features

- Upload CSV, XLSX, or XLS datasets.
- Inspect dataset shape, columns, dtypes, and preview rows.
- Choose a problem type: classification, regression, or clustering.
- Select a target column for supervised learning.
- Review backend-generated preprocessing recommendations before training.
- Use default algorithms or choose custom algorithms.
- Enable GridSearchCV and optionally provide custom parameter grids.
- Compare model performance across algorithms.
- View evaluation metrics, preprocessing reports, feature importance, confusion matrices, and clustering profiles.
- Download the trained model bundle for later use.

## Tech Stack

Backend:

- Python
- FastAPI
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- XGBoost
- joblib

Frontend:

- React
- TypeScript
- Vite
- Material UI
- Axios
- Recharts

## Project Structure

```text
.
├── data_eng_ml/
│   ├── app.py              # FastAPI endpoints
│   ├── ml_pipeline.py      # Data analysis, preprocessing, training, evaluation, export
│   ├── requirements.txt    # Backend dependencies
│   ├── sessions/           # Uploaded datasets and session metadata
│   └── saved_models/       # Exported model bundles
└── frontend/
    ├── src/
    │   ├── api/            # API client
    │   ├── components/     # Shared UI components
    │   ├── context/        # App state
    │   ├── pages/          # Upload, configure, review, results pages
    │   └── types/          # API response/request types
    ├── package.json
    └── vite.config.ts
```

## Backend API

The FastAPI service exposes these main endpoints:

- `GET /`: health check.
- `POST /upload`: uploads a dataset and creates a session.
- `POST /analyze`: analyzes the uploaded dataset and returns recommendation cards.
- `POST /train`: applies preprocessing choices, trains models, evaluates the best model, and saves the model bundle.
- `GET /download/{session_id}`: downloads the saved `.joblib` model bundle.

## How To Run Locally

### 1. Start the backend

From the project root:

```powershell
cd data_eng_ml
py -3.13 -m venv .venv-win
.\.venv-win\Scripts\python.exe -m ensurepip --upgrade --default-pip
.\.venv-win\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-win\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8010
```

Then open:

```text
http://127.0.0.1:8010/docs
```

Note: port `8000` may already be used on some machines. If it is free, you can use `--port 8000`. If you run the backend on `8010`, set the frontend API URL as shown below.

### 2. Start the frontend

Open a second terminal:

```powershell
cd frontend
npm install
$env:VITE_API_BASE_URL="http://127.0.0.1:8010"
npm run dev
```

Then open the Vite URL, usually:

```text
http://localhost:5173
```

If the backend is running on `8000`, the frontend can also use its default API URL without setting `VITE_API_BASE_URL`.
