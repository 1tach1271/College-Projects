"""
ml.py — Module 6: Machine Learning Layer
==========================================
Trains and evaluates models for:
  A. Market regime classification (3-class)
  B. Volatility spike prediction (binary)

Uses cuML (GPU) with scikit-learn (CPU) fallback.
Time-aware train/test split with purge gap to prevent data leakage.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

import config
from gpu_utils import is_gpu_available, gpu_context
from utils import timed

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ─── Feature Column Selection ───────────────────────────────────────────────

FEATURE_COLUMNS = [
    # Volatility features
    "vol_5d", "vol_10d", "vol_21d", "vol_63d", "vol_126d",
    # Returns & momentum
    "log_return", "momentum_5d", "momentum_10d", "momentum_21d",
    # Moving average signals
    "ma_crossover_5_63", "ma_crossover_10_126",
    # RSI
    "rsi_14",
    # Drawdown
    "drawdown", "max_drawdown_63d",
    # Liquidity
    "volume_zscore", "amihud_21d",
    # Intraday vol
    "parkinson_vol", "parkinson_vol_21d",
    # Cross-sectional
    "vol_rank", "return_rank", "market_breadth",
    # Spectral features
    "dominant_freq", "spectral_entropy",
    "low_freq_energy", "mid_freq_energy", "high_freq_energy",
    "spectral_edge_freq",
]


def _get_available_features(df):
    """Return feature columns that exist in the DataFrame."""
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    logger.info(f"Available feature columns: {len(available)}/{len(FEATURE_COLUMNS)}")
    return available


# ─── Time-Aware Train/Test Split ────────────────────────────────────────────

@timed
def time_split(df, target_col):
    """
    Time-aware train/test split with purge gap.
    - Train: first 80% of dates
    - Gap: PURGE_GAP_DAYS between train and test
    - Test: remaining dates after gap
    """
    df = df.sort_values("Date").reset_index(drop=True)

    # Remove rows with missing target
    df = df.dropna(subset=[target_col])

    unique_dates = sorted(df["Date"].unique())
    n_dates = len(unique_dates)
    split_idx = int(n_dates * config.TRAIN_RATIO)
    gap = config.PURGE_GAP_DAYS

    train_end_date = unique_dates[split_idx]
    test_start_date = unique_dates[min(split_idx + gap, n_dates - 1)]

    train_df = df[df["Date"] < train_end_date]
    test_df = df[df["Date"] >= test_start_date]

    logger.info(f"Split for '{target_col}': "
                f"Train={len(train_df):,} ({str(train_df['Date'].min())[:10]} to {str(train_end_date)[:10]}), "
                f"Test={len(test_df):,} ({str(test_start_date)[:10]} to {str(test_df['Date'].max())[:10]}), "
                f"Purge gap={gap} days")

    return train_df, test_df


# ─── Model Training ────────────────────────────────────────────────────────

def _prepare_xy(df, feature_cols, target_col):
    """Prepare feature matrix and target vector."""
    subset = df[feature_cols + [target_col]].dropna()
    X = subset[feature_cols].values.astype(np.float32)
    y = subset[target_col].values

    # Replace infinities
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y


def _train_logistic_regression(X_train, y_train, X_test, y_test, task="classification"):
    """Train Logistic Regression (cuML GPU or sklearn CPU)."""
    if is_gpu_available():
        from cuml.linear_model import LogisticRegression as cuLR
        import cupy as cp
        with gpu_context("LogisticRegression"):
            model = cuLR(max_iter=config.LR_PARAMS["max_iter"], C=config.LR_PARAMS["C"])
            model.fit(cp.asarray(X_train), cp.asarray(y_train))
            preds = cp.asnumpy(model.predict(cp.asarray(X_test)))
            try:
                proba = cp.asnumpy(model.predict_proba(cp.asarray(X_test)))
            except Exception:
                proba = None
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, C=1.0, random_state=config.RANDOM_STATE)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        try:
            proba = model.predict_proba(X_test)
        except Exception:
            proba = None

    return preds, proba, model


def _train_random_forest(X_train, y_train, X_test, y_test, task="classification"):
    """Train Random Forest (cuML GPU or sklearn CPU)."""
    if is_gpu_available():
        from cuml.ensemble import RandomForestClassifier as cuRF
        import cupy as cp
        with gpu_context("RandomForest"):
            model = cuRF(
                n_estimators=config.RF_PARAMS["n_estimators"],
                max_depth=config.RF_PARAMS["max_depth"],
                random_state=config.RANDOM_STATE,
            )
            model.fit(cp.asarray(X_train), cp.asarray(y_train.astype(np.int32)))
            preds = cp.asnumpy(model.predict(cp.asarray(X_test)))
            try:
                proba = cp.asnumpy(model.predict_proba(cp.asarray(X_test)))
            except Exception:
                proba = None
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=config.RANDOM_STATE, n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

    return preds, proba, model


def _train_xgboost(X_train, y_train, X_test, y_test, n_classes=2, task="classification"):
    """Train XGBoost with GPU acceleration."""
    import xgboost as xgb

    params = config.XGB_PARAMS.copy()
    if n_classes > 2:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes
        params["eval_metric"] = "mlogloss"
    else:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params.pop("n_estimators", 300),
        evals=[(dtest, "test")],
        verbose_eval=False,
    )

    proba_raw = model.predict(dtest)
    if n_classes > 2:
        proba = proba_raw
        preds = np.argmax(proba_raw, axis=1)
    else:
        proba = np.column_stack([1 - proba_raw, proba_raw])
        preds = (proba_raw > 0.5).astype(int)

    return preds, proba, model


# ─── Evaluation ─────────────────────────────────────────────────────────────

def _evaluate(y_true, y_pred, y_proba, model_name, task, n_classes):
    """Compute and log evaluation metrics."""
    results = {"model": model_name}

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    results["accuracy"] = round(acc, 4)
    results["f1_macro"] = round(f1_macro, 4)

    if y_proba is not None and n_classes == 2:
        try:
            roc = roc_auc_score(y_true, y_proba[:, 1])
            results["roc_auc"] = round(roc, 4)
        except Exception:
            results["roc_auc"] = None
    elif y_proba is not None and n_classes > 2:
        try:
            roc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            results["roc_auc"] = round(roc, 4)
        except Exception:
            results["roc_auc"] = None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    results["confusion_matrix"] = cm

    # Classification report
    report = classification_report(y_true, y_pred, zero_division=0)
    results["classification_report"] = report

    logger.info(f"[{model_name}] Accuracy={acc:.4f}, F1(macro)={f1_macro:.4f}, "
                f"ROC-AUC={results.get('roc_auc', 'N/A')}")

    return results


# ─── Feature Importance ─────────────────────────────────────────────────────

def _get_feature_importance(model, feature_cols, model_name):
    """Extract feature importance from the best model."""
    importance = None

    if "xgboost" in model_name.lower() or hasattr(model, "get_score"):
        try:
            scores = model.get_score(importance_type="gain")
            importance = pd.Series(
                {feature_cols[int(k.replace("f", ""))]: v for k, v in scores.items()}
            ).sort_values(ascending=False)
        except Exception:
            pass
    elif hasattr(model, "feature_importances_"):
        try:
            imp = model.feature_importances_
            if hasattr(imp, "get"):
                imp = imp.get()
            importance = pd.Series(imp, index=feature_cols).sort_values(ascending=False)
        except Exception:
            pass

    return importance


# ─── Main ML Pipeline ──────────────────────────────────────────────────────

@timed
def train_regime_classifier(df):
    """
    Task A: Market regime classification (3-class).
    """
    logger.info("─── Training Regime Classifier (3-class) ───")

    feature_cols = _get_available_features(df)
    target_col = "regime_label"

    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found!")
        return {}

    train_df, test_df = time_split(df, target_col)
    X_train, y_train = _prepare_xy(train_df, feature_cols, target_col)
    X_test, y_test = _prepare_xy(test_df, feature_cols, target_col)

    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("Insufficient data for training")
        return {}

    n_classes = len(np.unique(y_train))
    logger.info(f"Training data: {X_train.shape}, Test data: {X_test.shape}, Classes: {n_classes}")

    results = {}

    # Logistic Regression
    try:
        preds, proba, model = _train_logistic_regression(X_train, y_train, X_test, y_test)
        results["logistic_regression"] = _evaluate(y_test, preds, proba, "LogisticRegression", "regime", n_classes)
        results["logistic_regression"]["model"] = model
    except Exception as e:
        logger.warning(f"LogisticRegression failed: {e}")

    # Random Forest
    try:
        preds, proba, model = _train_random_forest(X_train, y_train, X_test, y_test)
        results["random_forest"] = _evaluate(y_test, preds, proba, "RandomForest", "regime", n_classes)
        results["random_forest"]["model"] = model
        fi = _get_feature_importance(model, feature_cols, "RandomForest")
        if fi is not None:
            results["random_forest"]["feature_importance"] = fi
    except Exception as e:
        logger.warning(f"RandomForest failed: {e}")

    # XGBoost
    try:
        preds, proba, model = _train_xgboost(X_train, y_train, X_test, y_test, n_classes)
        results["xgboost"] = _evaluate(y_test, preds, proba, "XGBoost", "regime", n_classes)
        results["xgboost"]["model"] = model
        fi = _get_feature_importance(model, feature_cols, "XGBoost")
        if fi is not None:
            results["xgboost"]["feature_importance"] = fi
    except Exception as e:
        logger.warning(f"XGBoost failed: {e}")

    results["feature_cols"] = feature_cols
    return results


@timed
def train_vol_spike_predictor(df):
    """
    Task B: Volatility spike prediction (binary).
    """
    logger.info("─── Training Volatility Spike Predictor (binary) ───")

    feature_cols = _get_available_features(df)
    target_col = "vol_spike"

    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found!")
        return {}

    train_df, test_df = time_split(df, target_col)
    X_train, y_train = _prepare_xy(train_df, feature_cols, target_col)
    X_test, y_test = _prepare_xy(test_df, feature_cols, target_col)

    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("Insufficient data for training")
        return {}

    # Class imbalance check
    spike_ratio = y_train.mean()
    logger.info(f"Vol spike prevalence in train: {spike_ratio:.3f}")

    results = {}

    # Logistic Regression
    try:
        preds, proba, model = _train_logistic_regression(X_train, y_train, X_test, y_test)
        results["logistic_regression"] = _evaluate(y_test, preds, proba, "LogisticRegression", "spike", 2)
        results["logistic_regression"]["model"] = model
    except Exception as e:
        logger.warning(f"LogisticRegression failed: {e}")

    # Random Forest
    try:
        preds, proba, model = _train_random_forest(X_train, y_train, X_test, y_test)
        results["random_forest"] = _evaluate(y_test, preds, proba, "RandomForest", "spike", 2)
        results["random_forest"]["model"] = model
    except Exception as e:
        logger.warning(f"RandomForest failed: {e}")

    # XGBoost (with scale_pos_weight for imbalance)
    try:
        import xgboost as xgb
        params = config.XGB_PARAMS.copy()
        params["objective"] = "binary:logistic"
        params["scale_pos_weight"] = max((1 - spike_ratio) / (spike_ratio + 1e-10), 1)
        n_rounds = params.pop("n_estimators", 300)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(params, dtrain, num_boost_round=n_rounds,
                          evals=[(dtest, "test")], verbose_eval=False)
        proba_raw = model.predict(dtest)
        proba = np.column_stack([1 - proba_raw, proba_raw])
        preds = (proba_raw > 0.5).astype(int)

        results["xgboost"] = _evaluate(y_test, preds, proba, "XGBoost", "spike", 2)
        results["xgboost"]["model"] = model
        fi = _get_feature_importance(model, feature_cols, "XGBoost")
        if fi is not None:
            results["xgboost"]["feature_importance"] = fi
    except Exception as e:
        logger.warning(f"XGBoost failed: {e}")

    results["feature_cols"] = feature_cols
    return results


@timed
def run_ml(df):
    """
    Full ML pipeline:
    1. Train regime classifier
    2. Train volatility spike predictor
    Returns: dict with both result sets
    """
    logger.info("═══ Starting Machine Learning Layer ═══")

    ml_results = {
        "regime": train_regime_classifier(df),
        "vol_spike": train_vol_spike_predictor(df),
    }

    # Summary
    for task_name, task_results in ml_results.items():
        if task_results:
            best_f1 = 0
            best_model = ""
            for model_name in ["logistic_regression", "random_forest", "xgboost"]:
                if model_name in task_results and "f1_macro" in task_results[model_name]:
                    f1 = task_results[model_name]["f1_macro"]
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = model_name
            logger.info(f"[{task_name}] Best model: {best_model} (F1={best_f1:.4f})")

    return ml_results
