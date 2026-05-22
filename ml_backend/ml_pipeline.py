# ml_pipeline.py

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — ANALYZE
# Inspect the uploaded dataframe and return all detected issues.
# The frontend shows these as yes/no prompt cards to the user.
# ──────────────────────────────────────────────────────────────────────────────

def analyze_data(df, task_type, target_col):
    feat_cols = [c for c in df.columns if c != target_col]
    X = df[feat_cols]
    issues = {}

    # Missing values
    missing_counts = {col: int(n) for col, n in X.isnull().sum().items() if n > 0}
    if missing_counts:
        total = sum(missing_counts.values())
        issues['missing_values'] = {
            'question': f"Found {total} missing values across {len(missing_counts)} column(s). Should I fix them?",
            'detail':   missing_counts,
            'options': {
                'strategy_numeric':      ['median', 'mean', 'constant_0'],
                'strategy_categorical':  ['most_frequent', 'constant_unknown'],
            },
            'defaults': {'strategy_numeric': 'median', 'strategy_categorical': 'most_frequent'},
        }

    # Duplicate rows
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        issues['duplicates'] = {
            'question': f"Found {n_dupes} duplicate row(s). Should I remove them?",
            'detail':   f"{n_dupes} duplicate rows",
        }

    # Categorical encoding
    cat_cols   = X.select_dtypes(include=['object', 'category']).columns.tolist()
    normal_cat = [c for c in cat_cols if X[c].nunique(dropna=True) <= 50]
    high_card  = [c for c in cat_cols if X[c].nunique(dropna=True) > 50]

    if normal_cat:
        issues['categorical_encoding'] = {
            'question': f"Found {len(normal_cat)} categorical column(s). Should I encode them?",
            'detail':   normal_cat,
            'options':  {'encoding_strategy': ['onehot', 'frequency_encode']},
            'defaults': {'encoding_strategy': 'onehot'},
        }

    if high_card:
        issues['high_cardinality'] = {
            'question': f"Columns {high_card} have >50 unique values. One-hot would create too many features. Drop them?",
            'detail':   high_card,
            'options':  {'strategy': ['drop', 'frequency_encode']},
            'defaults': {'strategy': 'drop'},
        }

    # Numerical scaling
    num_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    if num_cols:
        issues['scaling'] = {
            'question': f"Found {len(num_cols)} numerical feature(s). Should I scale them?",
            'detail':   num_cols,
            'options':  {'scaler': ['standard', 'minmax', 'robust']},
            'defaults': {'scaler': 'standard'},
        }

    # Outlier detection (numeric features)
    if num_cols:
        outlier_counts = {}
        k = 1.5
        for col in num_cols:
            series = X[col].dropna()
            if len(series) < 4:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            count = int(((series < lower) | (series > upper)).sum())
            if count > 0:
                outlier_counts[col] = count

        if outlier_counts:
            issues['outliers'] = {
                'question': f"Detected potential outliers in {len(outlier_counts)} numeric column(s). Apply IQR capping?",
                'detail': outlier_counts,
                'options': {
                    'outlier_method': ['iqr_cap'],
                    'outlier_iqr_k': [1.0, 1.5, 2.0, 3.0],
                },
                'defaults': {
                    'outlier_method': 'iqr_cap',
                    'outlier_iqr_k': 1.5,
                },
            }

    # Class imbalance — classification only
    if task_type == 'classification' and target_col:
        vc    = df[target_col].value_counts()
        ratio = float(vc.iloc[0] / vc.iloc[-1]) if len(vc) > 1 else 1.0
        if ratio > 1.5:
            issues['class_imbalance'] = {
                'question': f"Class imbalance detected (majority/minority ratio = {ratio:.1f}x). Should I resample?",
                'detail':   {'distribution': vc.to_dict(), 'ratio': round(ratio, 2)},
                'options':  {'resample_strategy': ['smote', 'random_oversample', 'random_undersample']},
                'defaults': {'resample_strategy': 'smote'},
            }

    # High dimensionality
    if len(feat_cols) > 100:
        issues['high_dimensionality'] = {
            'question': f"Dataset has {len(feat_cols)} features. Should I apply PCA to reduce dimensionality?",
            'detail':   f"{len(feat_cols)} features",
            'options':  {'n_components': 'integer or null for auto (95% variance)'},
            'defaults': {'n_components': None},
        }

    return issues


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — PREPROCESS
# Execute preprocessing based on the user's choices from the frontend.
# ──────────────────────────────────────────────────────────────────────────────

def run_preprocessing(df, task_type, target_col, choices):
    """
    choices is a flat dict of user answers, e.g.:
    {
        'duplicates':               True,
        'missing_values':           True,
        'strategy_numeric':         'median',
        'strategy_categorical':     'most_frequent',
        'categorical_encoding':     True,
        'encoding_strategy':        'onehot',
        'high_cardinality':         True,
        'high_cardinality_strategy':'drop',
        'scaling':                  True,
        'scaler':                   'standard',
        'class_imbalance':          True,
        'resample_strategy':        'smote',
        'high_dimensionality':      False,
        'pca_n_components':         None,
        'test_size':                0.2,
        'random_state':             42,
    }
    Returns: X_train, X_test, y_train, y_test, feature_names, artifacts, report
    """
    data         = df.copy()
    report       = []
    test_size    = choices.get('test_size', 0.2)
    random_state = choices.get('random_state', 42)
    preprocessors = {
        'numeric_imputer': None,
        'categorical_imputer': None,
        'outliers': None,
        'high_cardinality': None,
        'categorical_encoder': None,
        'scaler': None,
        'pca': None,
    }

    # Separate target from features
    if target_col:
        y_raw = data[target_col].copy()
        X     = data.drop(columns=[target_col])
    else:
        y_raw = None
        X     = data.copy()

    # ── Remove duplicates ──────────────────────────────────────────────
    if choices.get('duplicates', True):
        before = len(X)
        mask   = ~data.duplicated()
        X      = X[mask].reset_index(drop=True)
        if y_raw is not None:
            y_raw = y_raw[mask].reset_index(drop=True)
        if before - len(X) > 0:
            report.append(f"Removed {before - len(X)} duplicate rows.")

    # ── Encode target ──────────────────────────────────────────────────
    label_encoder = None
    y_processed   = None

    if task_type == 'classification' and y_raw is not None:
        le            = LabelEncoder()
        y_processed   = le.fit_transform(y_raw)
        label_encoder = le
        report.append(f"Label-encoded target '{target_col}': classes = {list(le.classes_)}.")
    elif task_type == 'regression' and y_raw is not None:
        y_processed = y_raw.values.astype(float)

    # ── Train / test split ─────────────────────────────────────────────
    if task_type in ('classification', 'regression'):
        stratify = None
        if task_type == 'classification':
            # Stratified split requires each class to have at least 2 samples.
            # If not possible, fall back to a normal random split.
            values, counts = np.unique(y_processed, return_counts=True)
            if len(values) > 1 and counts.min() >= 2:
                stratify = y_processed

        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X, y_processed, test_size=test_size, random_state=random_state, stratify=stratify
        )
        X_train_df = X_train_df.reset_index(drop=True)
        X_test_df  = X_test_df.reset_index(drop=True)
        if stratify is None and task_type == 'classification':
            report.append("Split: used non-stratified train/test split (class counts too small for stratification).")
        report.append(f"Split: {len(X_train_df)} train / {len(X_test_df)} test samples.")
    else:
        X_train_df = X.reset_index(drop=True)
        X_test_df  = None
        y_train = y_test = None
        report.append("Prepared the full dataset for clustering. No separate test split is used for unsupervised evaluation.")

    # ── Impute missing values ──────────────────────────────────────────
    if choices.get('missing_values', True):
        num_strategy = choices.get('strategy_numeric', 'median')
        cat_strategy = choices.get('strategy_categorical', 'most_frequent')

        num_cols = X_train_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        cat_cols = X_train_df.select_dtypes(include=['object', 'category']).columns.tolist()

        if num_cols:
            sk_strategy = 'constant' if num_strategy == 'constant_0' else num_strategy
            fill_value  = 0 if num_strategy == 'constant_0' else None
            imp = SimpleImputer(strategy=sk_strategy, fill_value=fill_value)
            X_train_df[num_cols] = imp.fit_transform(X_train_df[num_cols])
            if X_test_df is not None:
                X_test_df[num_cols] = imp.transform(X_test_df[num_cols])
            preprocessors['numeric_imputer'] = {
                'columns': num_cols,
                'transformer': imp,
                'strategy': num_strategy,
            }
            report.append(f"Imputed numeric columns with strategy='{num_strategy}'.")

        if cat_cols:
            sk_strategy = 'constant' if cat_strategy == 'constant_unknown' else cat_strategy
            fill_value  = 'Unknown' if cat_strategy == 'constant_unknown' else None
            imp = SimpleImputer(strategy=sk_strategy, fill_value=fill_value)
            X_train_df[cat_cols] = imp.fit_transform(X_train_df[cat_cols])
            if X_test_df is not None:
                X_test_df[cat_cols] = imp.transform(X_test_df[cat_cols])
            preprocessors['categorical_imputer'] = {
                'columns': cat_cols,
                'transformer': imp,
                'strategy': cat_strategy,
            }
            report.append(f"Imputed categorical columns with strategy='{cat_strategy}'.")

    # ── High-cardinality columns ───────────────────────────────────────
    hc_cols = [c for c in X_train_df.select_dtypes(include=['object', 'category']).columns
               if X_train_df[c].nunique(dropna=True) > 50]
    if hc_cols and choices.get('high_cardinality', True):
        hc_strategy = choices.get('high_cardinality_strategy', 'drop')
        if hc_strategy == 'drop':
            X_train_df = X_train_df.drop(columns=hc_cols)
            if X_test_df is not None:
                X_test_df = X_test_df.drop(columns=hc_cols)
            preprocessors['high_cardinality'] = {'strategy': 'drop', 'columns': hc_cols}
            report.append(f"Dropped {len(hc_cols)} high-cardinality column(s): {hc_cols}.")
        else:
            mappings = {}
            for col in hc_cols:
                freq_map = X_train_df[col].value_counts(normalize=True, dropna=False).to_dict()
                mappings[col] = freq_map
                X_train_df[col] = X_train_df[col].map(freq_map).fillna(0.0).astype(float)
                if X_test_df is not None:
                    X_test_df[col] = X_test_df[col].map(freq_map).fillna(0.0).astype(float)
            preprocessors['high_cardinality'] = {
                'strategy': 'frequency_encode',
                'columns': hc_cols,
                'mappings': mappings,
            }
            report.append(f"Frequency-encoded {len(hc_cols)} high-cardinality column(s).")

    # ── Encode categorical columns ─────────────────────────────────────
    cat_cols     = X_train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    enc_strategy = choices.get('encoding_strategy', 'onehot')

    if choices.get('categorical_encoding', True) and cat_cols:
        if enc_strategy == 'onehot':
            ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            train_encoded = ohe.fit_transform(X_train_df[cat_cols])
            enc_names = ohe.get_feature_names_out(cat_cols).tolist()
            train_encoded_df = pd.DataFrame(train_encoded, columns=enc_names, index=X_train_df.index)
            X_train_df = pd.concat([X_train_df.drop(columns=cat_cols), train_encoded_df], axis=1)

            if X_test_df is not None:
                test_encoded = ohe.transform(X_test_df[cat_cols])
                test_encoded_df = pd.DataFrame(test_encoded, columns=enc_names, index=X_test_df.index)
                X_test_df = pd.concat([X_test_df.drop(columns=cat_cols), test_encoded_df], axis=1)

            preprocessors['categorical_encoder'] = {
                'strategy': 'onehot',
                'columns': cat_cols,
                'transformer': ohe,
                'feature_names': enc_names,
            }
            report.append(f"One-hot encoded {len(cat_cols)} column(s) -> {len(enc_names)} binary features.")
        else:
            mappings = {}
            for col in cat_cols:
                freq_map = X_train_df[col].value_counts(normalize=True, dropna=False).to_dict()
                mappings[col] = freq_map
                X_train_df[col] = X_train_df[col].map(freq_map).fillna(0.0).astype(float)
                if X_test_df is not None:
                    X_test_df[col] = X_test_df[col].map(freq_map).fillna(0.0).astype(float)
            preprocessors['categorical_encoder'] = {
                'strategy': 'frequency_encode',
                'columns': cat_cols,
                'mappings': mappings,
            }
            report.append(f"Frequency-encoded {len(cat_cols)} categorical column(s).")

    # ── Outlier handling (IQR capping) ─────────────────────────────────
    if choices.get('outliers', task_type in ('classification', 'regression')):
        outlier_method = choices.get('outlier_method', 'iqr_cap')
        try:
            outlier_iqr_k = float(choices.get('outlier_iqr_k', 1.5))
        except (TypeError, ValueError):
            outlier_iqr_k = 1.5

        if outlier_method == 'iqr_cap':
            outlier_cols = X_train_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
            bounds = {}
            capped_summary = {}
            for col in outlier_cols:
                train_series = X_train_df[col]
                non_null = train_series.dropna()
                if len(non_null) < 4:
                    continue
                q1 = non_null.quantile(0.25)
                q3 = non_null.quantile(0.75)
                iqr = q3 - q1
                if pd.isna(iqr) or iqr == 0:
                    continue

                lower = float(q1 - outlier_iqr_k * iqr)
                upper = float(q3 + outlier_iqr_k * iqr)
                bounds[col] = {'lower': lower, 'upper': upper}

                train_before = int(((X_train_df[col] < lower) | (X_train_df[col] > upper)).sum())
                X_train_df[col] = X_train_df[col].clip(lower=lower, upper=upper)

                test_before = 0
                if X_test_df is not None:
                    test_before = int(((X_test_df[col] < lower) | (X_test_df[col] > upper)).sum())
                    X_test_df[col] = X_test_df[col].clip(lower=lower, upper=upper)

                total = train_before + test_before
                if total > 0:
                    capped_summary[col] = {'train': train_before, 'test': test_before}

            preprocessors['outliers'] = {
                'method': 'iqr_cap',
                'k': outlier_iqr_k,
                'columns': list(bounds.keys()),
                'bounds': bounds,
            }
            if capped_summary:
                report.append(
                    f"Applied IQR capping (k={outlier_iqr_k}) on {len(capped_summary)} column(s): {capped_summary}."
                )
            else:
                report.append(f"Outlier handling enabled (IQR cap, k={outlier_iqr_k}), but no values needed capping.")

    # ── Scale numerical features ───────────────────────────────────────
    num_cols      = X_train_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    scaler_choice = choices.get('scaler', 'standard')

    if choices.get('scaling', True) and num_cols:
        scalers = {'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'robust': RobustScaler()}
        scaler  = scalers.get(scaler_choice, StandardScaler())
        X_train_df[num_cols] = scaler.fit_transform(X_train_df[num_cols])
        if X_test_df is not None:
            X_test_df[num_cols] = scaler.transform(X_test_df[num_cols])
        preprocessors['scaler'] = {
            'columns': num_cols,
            'transformer': scaler,
            'strategy': scaler_choice,
        }
        report.append(f"Applied {scaler_choice} scaling to {len(num_cols)} numerical column(s).")

    feature_names = X_train_df.columns.tolist()
    X_train       = X_train_df.values.astype(float)
    X_test        = X_test_df.values.astype(float) if X_test_df is not None else X_train.copy()

    # ── Resample training set (classification only) ────────────────────
    if task_type == 'classification' and choices.get('class_imbalance', False):
        strategy = choices.get('resample_strategy', 'smote')
        if strategy == 'smote':
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=random_state)
        elif strategy == 'random_oversample':
            from imblearn.over_sampling import RandomOverSampler
            sampler = RandomOverSampler(random_state=random_state)
        else:
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(random_state=random_state)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        report.append(f"Resampled training set with '{strategy}'. New train size: {len(X_train)}.")

    # ── PCA ────────────────────────────────────────────────────────────
    if choices.get('high_dimensionality', False):
        n_comp  = choices.get('pca_n_components', None)
        pca     = PCA(n_components=n_comp if n_comp else 0.95, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        if X_test is not None:
            X_test = pca.transform(X_test)
        feature_names = [f'PC{i+1}' for i in range(X_train.shape[1])]
        preprocessors['pca'] = pca
        report.append(f"PCA applied: {X_train.shape[1]} components retained (95% variance threshold).")

    artifacts = {
        'label_encoder': label_encoder,
        'preprocessing': preprocessors,
    }

    return X_train, X_test, y_train, y_test, feature_names, artifacts, report


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — TRAIN
# Train all selected algorithms, pick best by CV score.
# ──────────────────────────────────────────────────────────────────────────────

# Default hyperparameter search spaces
CLF_PARAMS = {
    'logistic_regression': {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']},
    'random_forest':       {'n_estimators': [100, 200], 'max_depth': [5, 10, None], 'max_features': ['sqrt', 'log2']},
    'svm':                 {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'xgboost':             {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
}

REG_PARAMS = {
    'linear_regression': {},
    'ridge':             {'alpha': [0.01, 0.1, 1, 10, 100]},
    'lasso':             {'alpha': [0.01, 0.1, 1, 10]},
    'random_forest':     {'n_estimators': [100, 200], 'max_depth': [5, 10, None], 'max_features': ['sqrt', 'log2']},
}

CLUS_PARAMS = {
    'kmeans': {
        'n_clusters': list(range(2, 9)),
    },
    'dbscan': {
        'eps': [0.3, 0.5, 0.7, 1.0],
        'min_samples': [3, 5, 10],
    },
}


def _make_clf(name, random_state):
    if name == 'logistic_regression':
        return LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state)
    if name == 'random_forest':
        return RandomForestClassifier(class_weight='balanced', random_state=random_state)
    if name == 'svm':
        return SVC(probability=True, class_weight='balanced', random_state=random_state)
    if name == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(eval_metric='logloss', verbosity=0, random_state=random_state)
    raise ValueError(f"Unknown classifier: {name}")


def _make_reg(name, random_state):
    if name == 'linear_regression':
        return LinearRegression()
    if name == 'ridge':
        return Ridge()
    if name == 'lasso':
        return Lasso(max_iter=5000)
    if name == 'random_forest':
        return RandomForestRegressor(random_state=random_state)
    raise ValueError(f"Unknown regressor: {name}")


def build_cluster_profiles_and_labels(X_eval, labels, feature_names):
    """
    Build rule-based cluster labels from numeric differentiators.
    Returns:
      cluster_labels_map: {cluster_id: "Label text"}
      cluster_profiles: {cluster_id: {top_numeric_signals: [...], dominant_categories: []}}
      cluster_sizes: {cluster_id: int}
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        labels = labels.ravel()

    df = pd.DataFrame(X_eval, columns=feature_names)
    global_means = df.mean(numeric_only=True)
    global_stds = df.std(numeric_only=True).replace(0, np.nan)

    cluster_labels_map = {}
    cluster_profiles = {}
    cluster_sizes = {}

    for cluster_id in sorted(np.unique(labels)):
        cluster_mask = labels == cluster_id
        cluster_df = df.loc[cluster_mask]
        cluster_sizes[str(cluster_id)] = int(cluster_mask.sum())

        if cluster_id == -1:
            cluster_labels_map[str(cluster_id)] = "Noise / Outliers"
            cluster_profiles[str(cluster_id)] = {
                "top_numeric_signals": ["Sparse density region"],
                "dominant_categories": [],
            }
            continue

        if cluster_df.empty:
            cluster_labels_map[str(cluster_id)] = f"Cluster {cluster_id}"
            cluster_profiles[str(cluster_id)] = {
                "top_numeric_signals": ["No profile available"],
                "dominant_categories": [],
            }
            continue

        cluster_means = cluster_df.mean(numeric_only=True)
        z_scores = ((cluster_means - global_means) / global_stds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        top_features = z_scores.abs().sort_values(ascending=False).head(3)

        signals = []
        label_tokens = []
        for feat in top_features.index:
            z = float(z_scores.loc[feat])
            direction = "High" if z >= 0 else "Low"
            feat_name = feat.replace("_", " ").title()
            signals.append(f"{direction} {feat_name} (z={z:.2f})")
            label_tokens.append(f"{direction} {feat_name}")

        if not label_tokens:
            label_tokens = [f"Cluster {cluster_id}"]
            signals = ["No strong differentiators"]

        cluster_labels_map[str(cluster_id)] = ", ".join(label_tokens[:2])
        cluster_profiles[str(cluster_id)] = {
            "top_numeric_signals": signals,
            "dominant_categories": [],
        }

    return cluster_labels_map, cluster_profiles, cluster_sizes


def train_models(X_train, y_train, task_type, selected_algos, use_grid_search,
                 custom_params=None, random_state=42):
    """
    Train all selected algorithms. For supervised tasks, uses GridSearchCV if
    use_grid_search=True. For clustering, searches over a candidate list and
    picks the config with the best silhouette score.

    Returns:
        results     dict  { algo_name: { model, cv_score, best_params } }
        best_model  fitted estimator
        best_name   str
        best_score  float
    """
    results    = {}
    best_model = None
    best_name  = None
    best_score = -np.inf

    # ── Classification ─────────────────────────────────────────────────
    if task_type == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        for name in selected_algos:
            try:
                model  = _make_clf(name, random_state)
                params = (custom_params or {}).get(name, CLF_PARAMS[name])

                if use_grid_search and params:
                    gs = GridSearchCV(model, params, scoring='f1_weighted', cv=cv, n_jobs=-1)
                    gs.fit(X_train, y_train)
                    fitted      = gs.best_estimator_
                    cv_score    = round(gs.best_score_, 4)
                    best_params = gs.best_params_
                else:
                    model.fit(X_train, y_train)
                    fitted      = model
                    cv_score    = None
                    best_params = {}

                results[name] = {'model': fitted, 'cv_score': cv_score, 'best_params': best_params}
                print(f"  {name}: cv_score={cv_score}, params={best_params}")

                # When grid search is off, cv_score is None — use a small positive constant
                # so the first algorithm always becomes the initial best_model
                score = cv_score if cv_score is not None else (0 if best_model is not None else 1)
                if score > best_score:
                    best_score = score
                    best_model = fitted
                    best_name  = name
            except Exception as exc:
                results[name] = {'error': f"{type(exc).__name__}: {exc}"}
                print(f"  {name}: failed -> {type(exc).__name__}: {exc}")

    # ── Regression ─────────────────────────────────────────────────────
    elif task_type == 'regression':
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

        for name in selected_algos:
            try:
                model  = _make_reg(name, random_state)
                params = (custom_params or {}).get(name, REG_PARAMS[name])

                if use_grid_search and params:
                    gs = GridSearchCV(model, params, scoring='r2', cv=cv, n_jobs=-1)
                    gs.fit(X_train, y_train)
                    fitted      = gs.best_estimator_
                    cv_score    = round(gs.best_score_, 4)
                    best_params = gs.best_params_
                else:
                    model.fit(X_train, y_train)
                    fitted      = model
                    cv_score    = None
                    best_params = {}

                results[name] = {'model': fitted, 'cv_score': cv_score, 'best_params': best_params}
                print(f"  {name}: cv_score={cv_score}, params={best_params}")

                score = cv_score if cv_score is not None else (0 if best_model is not None else 1)
                if score > best_score:
                    best_score = score
                    best_model = fitted
                    best_name  = name
            except Exception as exc:
                results[name] = {'error': f"{type(exc).__name__}: {exc}"}
                print(f"  {name}: failed -> {type(exc).__name__}: {exc}")

    # ── Clustering ─────────────────────────────────────────────────────
    elif task_type == 'clustering':
        selected_algos = [name for name in selected_algos if name in ('kmeans', 'dbscan')]
        if not selected_algos:
            return {}, None, None, None
        # Explicitly keep KMeans k automatic via silhouette over CLUS_PARAMS range.
        # custom_params is intentionally ignored for clustering k selection.
        candidates = {
            'kmeans': [KMeans(n_clusters=k, random_state=random_state, n_init=10)
                       for k in CLUS_PARAMS['kmeans']['n_clusters']],
            'dbscan': [DBSCAN(eps=e, min_samples=m)
                       for e in CLUS_PARAMS['dbscan']['eps']
                       for m in CLUS_PARAMS['dbscan']['min_samples']],
        }

        for name in selected_algos:
            best_k      = None
            best_labels = None
            best_sil    = -1.0
            best_cfg    = {}

            for model in candidates[name]:
                labels  = model.fit_predict(X_train)
                n_clust = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clust < 2:
                    continue
                sil = silhouette_score(X_train, labels, sample_size=min(5000, len(X_train)))
                if sil > best_sil:
                    best_sil = sil
                    best_k   = model
                    best_labels = labels
                    best_cfg = model.get_params()

            if best_k is None:
                results[name] = {'error': 'No valid clustering configuration found.'}
                continue

            try:
                best_k._trained_labels = np.asarray(best_labels) if best_labels is not None else None
            except Exception:
                pass

            results[name] = {
                'model': best_k,
                'cv_score': round(best_sil, 4),
                'best_params': best_cfg,
                'labels': best_labels.tolist() if best_labels is not None else None,
            }
            print(f"  {name}: best_silhouette={best_sil:.4f}, params={best_cfg}")

            if best_sil > best_score:
                best_score = best_sil
                best_model = best_k
                best_name  = name

    return results, best_model, best_name, round(best_score, 4)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — EVALUATE
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, task_type, feature_names=None):
    if task_type == 'classification':
        y_pred = model.predict(X_test)
        return {
            'accuracy':           round(float(accuracy_score(y_test, y_pred)), 4),
            'precision_weighted': round(float(precision_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
            'recall_weighted':    round(float(recall_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
            'f1_weighted':        round(float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
            'f1_macro':           round(float(f1_score(y_test, y_pred, average='macro', zero_division=0)), 4),
            'confusion_matrix':   confusion_matrix(y_test, y_pred).tolist(),
        }

    elif task_type == 'regression':
        y_pred = model.predict(X_test)
        mse    = float(mean_squared_error(y_test, y_pred))
        return {
            'mae':  round(float(mean_absolute_error(y_test, y_pred)), 4),
            'mse':  round(mse, 4),
            'rmse': round(float(np.sqrt(mse)), 4),
            'r2':   round(float(r2_score(y_test, y_pred)), 4),
        }

    elif task_type == 'clustering':
        X_eval = X_test
        labels = getattr(model, '_trained_labels', None)
        if labels is None:
            if hasattr(model, 'predict'):
                labels = model.predict(X_eval)
            else:
                labels = model.fit_predict(X_eval)
        labels = np.asarray(labels)
        n_clust = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clust < 2:
            return {'error': 'Fewer than 2 clusters found in the clustering dataset.'}

        cluster_labels_map = {}
        cluster_profiles = {}
        cluster_sizes = {}
        if feature_names is not None:
            cluster_labels_map, cluster_profiles, cluster_sizes = build_cluster_profiles_and_labels(
                X_eval, labels, feature_names
            )

        return {
            'silhouette_score':        round(float(silhouette_score(X_eval, labels)), 4),
            'davies_bouldin_score':    round(float(davies_bouldin_score(X_eval, labels)), 4),
            'calinski_harabasz_score': round(float(calinski_harabasz_score(X_eval, labels)), 4),
            'n_clusters':              n_clust,
            'noise_points':            int((labels == -1).sum()),
            'cluster_labels_map':      cluster_labels_map,
            'cluster_profiles':        cluster_profiles,
            'cluster_sizes':           cluster_sizes,
        }

    return {}


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_feature_importance(model, feature_names):
    """Return top-20 feature importances, or None if model doesn't support it."""
    if hasattr(model, 'feature_importances_'):
        pairs = sorted(zip(feature_names, model.feature_importances_),
                       key=lambda x: x[1], reverse=True)
        return [{'feature': f, 'importance': round(float(v), 6)} for f, v in pairs[:20]]

    if hasattr(model, 'coef_'):
        coef = model.coef_
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)
        else:
            coef = np.abs(coef)
        pairs = sorted(zip(feature_names, coef), key=lambda x: x[1], reverse=True)
        return [{'feature': f, 'importance': round(float(v), 6)} for f, v in pairs[:20]]
    return None


def save_model(model, feature_names, task_type, target_col, artifacts, path):
    """Save model + metadata as a .pkl file. Load with joblib.load(path)."""
    if isinstance(artifacts, dict) and ('label_encoder' in artifacts or 'preprocessing' in artifacts):
        bundle_artifacts = artifacts
    else:
        bundle_artifacts = {'label_encoder': artifacts, 'preprocessing': {}}

    joblib.dump({
        'model':          model,
        'feature_names':  feature_names,
        'task_type':      task_type,
        'target_col':     target_col,
        'label_encoder':  bundle_artifacts.get('label_encoder'),
        'preprocessing':  bundle_artifacts.get('preprocessing', {}),
    }, path)
