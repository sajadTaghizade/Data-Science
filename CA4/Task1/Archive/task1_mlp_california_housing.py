"""
Task 1 - MLP regression on the California Housing dataset.

Run from the Task1 directory or the project root:
    python Task1/task1_mlp_california_housing.py

The script reads ../california_housing.csv read-only and writes all artifacts
inside Task1/.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
TARGET = "MedHouseVal"
ORIGINAL_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


ROOT_DIR = Path(__file__).resolve().parents[1]
TASK_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "california_housing.csv"
FIGURE_DIR = TASK_DIR / "figures"
MODEL_DIR = TASK_DIR / "models"
RESULTS_PATH = TASK_DIR / "metrics.json"
REPORT_PATH = TASK_DIR / "Task1_Report.md"


def set_reproducibility(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_current_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def markdown_table(rows: list[dict[str, object]], columns: Iterable[str]) -> str:
    columns = list(columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, separator, *body])


def format_float(value: float, ndigits: int = 4) -> str:
    return f"{value:.{ndigits}f}"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Expected dataset at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def compute_eda(df: pd.DataFrame) -> dict[str, object]:
    features = [col for col in df.columns if col != TARGET]
    summary = df.describe().T
    summary.to_csv(TASK_DIR / "eda_summary.csv")

    skewness = df[features + [TARGET]].skew().sort_values(ascending=False)
    skewness.to_csv(TASK_DIR / "eda_skewness.csv", header=["skewness"])

    missing = df.isna().sum()
    missing.to_csv(TASK_DIR / "eda_missing_values.csv", header=["missing_count"])

    q1 = df[features + [TARGET]].quantile(0.25)
    q3 = df[features + [TARGET]].quantile(0.75)
    iqr = q3 - q1
    outlier_counts = ((df[features + [TARGET]] < (q1 - 1.5 * iqr)) | (df[features + [TARGET]] > (q3 + 1.5 * iqr))).sum()
    outlier_counts.sort_values(ascending=False).to_csv(TASK_DIR / "eda_iqr_outlier_counts.csv", header=["iqr_outliers"])

    corr = df[features + [TARGET]].corr(numeric_only=True)
    target_corr = corr[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False)
    strong_pairs = []
    for i, left in enumerate(features):
        for right in features[i + 1 :]:
            value = corr.loc[left, right]
            if abs(value) >= 0.70:
                strong_pairs.append({"feature_1": left, "feature_2": right, "correlation": float(value)})

    capped_count = int((df[TARGET] >= 5.0).sum())

    return {
        "shape": df.shape,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_total": int(missing.sum()),
        "summary": summary,
        "skewness": skewness,
        "outlier_counts": outlier_counts.sort_values(ascending=False),
        "target_correlations": target_corr,
        "strong_feature_pairs": strong_pairs,
        "target_capped_count": capped_count,
        "target_capped_pct": capped_count / len(df) * 100,
    }


def create_eda_plots(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    features = [col for col in df.columns if col != TARGET]

    axes = df[features + [TARGET]].hist(figsize=(14, 11), bins=40, color="#356f8c", edgecolor="white")
    for ax in np.ravel(axes):
        ax.set_xlabel(ax.get_title())
        ax.set_ylabel("Count")
    save_current_figure("eda_histograms.png")

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, col in zip(axes.ravel(), features + [TARGET]):
        sns.boxplot(x=df[col], ax=ax, color="#d98b5f", fliersize=1.8)
        ax.set_title(f"{col} box plot")
        ax.set_xlabel(col)
    save_current_figure("eda_boxplots.png")

    plt.figure(figsize=(10, 8))
    corr = df[features + [TARGET]].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, linewidths=0.4)
    plt.title("Correlation heatmap for original features and target")
    save_current_figure("eda_correlation_heatmap.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    sns.scatterplot(data=df, x="MedInc", y=TARGET, alpha=0.25, s=12, ax=axes[0, 0], color="#2d6f7d")
    axes[0, 0].set_title("Median income vs median house value")

    sns.scatterplot(data=df, x="AveRooms", y=TARGET, alpha=0.25, s=12, ax=axes[0, 1], color="#8f5a83")
    axes[0, 1].set_xlim(0, df["AveRooms"].quantile(0.99))
    axes[0, 1].set_title("Average rooms vs median house value")

    sns.scatterplot(data=df, x="AveOccup", y=TARGET, alpha=0.25, s=12, ax=axes[1, 0], color="#6e7f36")
    axes[1, 0].set_xlim(0, df["AveOccup"].quantile(0.99))
    axes[1, 0].set_title("Average occupancy vs median house value")

    geo = axes[1, 1].scatter(
        df["Longitude"],
        df["Latitude"],
        c=df[TARGET],
        cmap="viridis",
        s=6,
        alpha=0.65,
    )
    axes[1, 1].set_title("Geographic pattern of median house value")
    axes[1, 1].set_xlabel("Longitude")
    axes[1, 1].set_ylabel("Latitude")
    fig.colorbar(geo, ax=axes[1, 1], label=TARGET)
    save_current_figure("eda_feature_target_relationships.png")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    eps = 1e-6

    engineered["RoomsPerPerson"] = engineered["AveRooms"] / (engineered["AveOccup"] + eps)
    engineered["BedroomsPerRoom"] = engineered["AveBedrms"] / (engineered["AveRooms"] + eps)
    engineered["RoomsPerBedroom"] = engineered["AveRooms"] / (engineered["AveBedrms"] + eps)
    engineered["IncomePerOccupant"] = engineered["MedInc"] / (engineered["AveOccup"] + eps)
    engineered["PopulationLog"] = np.log1p(engineered["Population"])
    engineered["AveOccupLog"] = np.log1p(engineered["AveOccup"])
    engineered["AveRoomsLog"] = np.log1p(engineered["AveRooms"])
    engineered["LatitudeLongitude"] = engineered["Latitude"] * engineered["Longitude"]
    engineered["LatitudeSquared"] = engineered["Latitude"] ** 2
    engineered["LongitudeSquared"] = engineered["Longitude"] ** 2

    # Approximate distances in degrees to two major high-value coastal job centers.
    engineered["DistanceToLosAngeles"] = np.sqrt((engineered["Latitude"] - 34.0522) ** 2 + (engineered["Longitude"] + 118.2437) ** 2)
    engineered["DistanceToSanFrancisco"] = np.sqrt((engineered["Latitude"] - 37.7749) ** 2 + (engineered["Longitude"] + 122.4194) ** 2)
    engineered["ClosestMajorCityDistance"] = engineered[["DistanceToLosAngeles", "DistanceToSanFrancisco"]].min(axis=1)

    return engineered


@dataclass
class SplitData:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    feature_names: list[str]


def prepare_split(
    df: pd.DataFrame,
    feature_names: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> SplitData:
    x = df[feature_names].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx]).astype(np.float32)
    x_val = scaler.transform(x[val_idx]).astype(np.float32)
    x_test = scaler.transform(x[test_idx]).astype(np.float32)

    return SplitData(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx],
        scaler=scaler,
        feature_names=feature_names,
    )


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float = 0.10) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def make_loaders(split: SplitData, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = TensorDataset(torch.from_numpy(split.x_train), torch.from_numpy(split.y_train))
    val_ds = TensorDataset(torch.from_numpy(split.x_val), torch.from_numpy(split.y_val))
    test_ds = TensorDataset(torch.from_numpy(split.x_test), torch.from_numpy(split.y_test))

    generator = torch.Generator()
    generator.manual_seed(SEED)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def epoch_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            losses.append(loss.item() * len(xb))
    return float(np.sum(losses) / len(loader.dataset))


def train_model(
    name: str,
    split: SplitData,
    hidden_dims: tuple[int, ...],
    lr: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 300,
    patience: int = 30,
    weight_decay: float = 1e-4,
    dropout: float = 0.10,
) -> tuple[MLPRegressor, dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(len(split.feature_names), hidden_dims=hidden_dims, dropout=dropout).to(device)
    train_loader, val_loader, _ = make_loaders(split, batch_size=batch_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = MODEL_DIR / f"{name}_best.pt"

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss = epoch_loss(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_names": split.feature_names,
                    "hidden_dims": hidden_dims,
                    "dropout": dropout,
                    "val_loss": best_val_loss,
                    "epoch": best_epoch,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metadata = {
        "name": name,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_run": len(history["epoch"]),
        "checkpoint": str(checkpoint_path.relative_to(TASK_DIR)),
        "device": str(device),
        "hidden_dims": hidden_dims,
        "learning_rate": lr,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "weight_decay": weight_decay,
        "dropout": dropout,
    }
    return model, metadata


def plot_training_curves(metadata: dict[str, object]) -> None:
    history = metadata["history"]
    epochs = history["epoch"]

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history["train_loss"], label="Training MSE", color="#2e6f95")
    plt.plot(epochs, history["val_loss"], label="Validation MSE", color="#c8563d")
    plt.axvline(metadata["best_epoch"], color="#333333", linestyle="--", linewidth=1.4, label=f"Best epoch {metadata['best_epoch']}")
    plt.title(f"{metadata['name'].title()} MLP training curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_current_figure(f"training_curve_{metadata['name']}.png")


def predict(model: nn.Module, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    device = next(model.parameters()).device
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())
    return np.vstack(preds).reshape(-1)


def evaluate_model(model: nn.Module, split: SplitData) -> dict[str, float]:
    y_true = split.y_test.reshape(-1)
    y_pred = predict(model, split.x_test)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
        "MSE": float(mse),
    }


def create_error_analysis_plots(
    best_name: str,
    best_model: nn.Module,
    split: SplitData,
    test_indices: np.ndarray,
    source_df: pd.DataFrame,
) -> dict[str, object]:
    y_true = split.y_test.reshape(-1)
    y_pred = predict(best_model, split.x_test)
    residuals = y_pred - y_true
    abs_error = np.abs(residuals)

    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.35, s=18, color="#2d6f7d")
    limit_min = min(float(y_true.min()), float(y_pred.min()))
    limit_max = max(float(y_true.max()), float(y_pred.max()))
    plt.plot([limit_min, limit_max], [limit_min, limit_max], color="#333333", linestyle="--", label="Perfect prediction")
    plt.title(f"{best_name.title()} model: predicted vs actual values")
    plt.xlabel("Actual MedHouseVal")
    plt.ylabel("Predicted MedHouseVal")
    plt.legend()
    save_current_figure("error_actual_vs_predicted.png")

    plt.figure(figsize=(9, 5))
    sns.histplot(residuals, bins=50, kde=True, color="#8f5a83")
    plt.axvline(0, color="#333333", linestyle="--", linewidth=1.2)
    plt.title(f"{best_name.title()} model residual distribution")
    plt.xlabel("Prediction error (predicted - actual)")
    plt.ylabel("Count")
    save_current_figure("error_residual_distribution.png")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=residuals, alpha=0.35, s=18, color="#b5792f")
    plt.axhline(0, color="#333333", linestyle="--", linewidth=1.2)
    plt.title(f"{best_name.title()} model residuals by actual value")
    plt.xlabel("Actual MedHouseVal")
    plt.ylabel("Residual")
    save_current_figure("error_residuals_by_actual.png")

    worst_order = np.argsort(abs_error)[-20:][::-1]
    worst_df = source_df.iloc[test_indices[worst_order]].copy()
    worst_df["ActualMedHouseVal"] = y_true[worst_order]
    worst_df["PredictedMedHouseVal"] = y_pred[worst_order]
    worst_df["Residual"] = residuals[worst_order]
    worst_df["AbsoluteError"] = abs_error[worst_order]
    worst_df.to_csv(TASK_DIR / "worst_test_predictions.csv", index=False)

    high_value_mask = y_true >= 4.5
    low_value_mask = y_true < 1.0
    capped_mask = y_true >= 5.0

    return {
        "mean_residual": float(np.mean(residuals)),
        "median_absolute_error": float(np.median(abs_error)),
        "p90_absolute_error": float(np.quantile(abs_error, 0.90)),
        "p95_absolute_error": float(np.quantile(abs_error, 0.95)),
        "high_value_mae": float(mean_absolute_error(y_true[high_value_mask], y_pred[high_value_mask])) if high_value_mask.any() else None,
        "low_value_mae": float(mean_absolute_error(y_true[low_value_mask], y_pred[low_value_mask])) if low_value_mask.any() else None,
        "capped_target_count_test": int(capped_mask.sum()),
        "worst_prediction_file": "worst_test_predictions.csv",
    }


def plot_model_comparison(metrics: dict[str, dict[str, float]], histories: dict[str, dict[str, object]]) -> None:
    metric_df = pd.DataFrame(metrics).T.reset_index(names="model")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, metric, color in zip(axes, ["MAE", "RMSE", "R2"], ["#2e6f95", "#c8563d", "#6e7f36"]):
        sns.barplot(data=metric_df, x="model", y=metric, ax=ax, color=color)
        ax.set_title(f"Test {metric}")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
    save_current_figure("model_comparison_metrics.png")

    plt.figure(figsize=(9, 6))
    for name, metadata in histories.items():
        history = metadata["history"]
        plt.plot(history["epoch"], history["val_loss"], label=f"{name} validation MSE")
    plt.title("Validation loss comparison")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_current_figure("model_comparison_validation_loss.png")


def make_report(
    df: pd.DataFrame,
    eda: dict[str, object],
    split_counts: dict[str, int],
    baseline_meta: dict[str, object],
    enhanced_meta: dict[str, object],
    metrics: dict[str, dict[str, float]],
    error_analysis: dict[str, object],
    engineered_features: list[str],
    best_name: str,
) -> None:
    top_skew = eda["skewness"].head(5)
    top_outliers = eda["outlier_counts"].head(5)
    target_corr = eda["target_correlations"]

    metric_rows = []
    for name, model_metrics in metrics.items():
        row = {"Model": name.title()}
        row.update({metric: format_float(value) for metric, value in model_metrics.items() if metric != "MSE"})
        row["Best epoch"] = baseline_meta["best_epoch"] if name == "baseline" else enhanced_meta["best_epoch"]
        row["Epochs run"] = baseline_meta["epochs_run"] if name == "baseline" else enhanced_meta["epochs_run"]
        metric_rows.append(row)

    feature_rows = [{"Engineered feature": feature} for feature in engineered_features]
    split_rows = [{"Split": split, "Rows": count, "Percent": format_float(count / len(df) * 100, 2)} for split, count in split_counts.items()]
    corr_rows = [{"Feature": idx, "Correlation with target": format_float(val)} for idx, val in target_corr.items()]
    skew_rows = [{"Feature": idx, "Skewness": format_float(val)} for idx, val in top_skew.items()]
    outlier_rows = [{"Feature": idx, "IQR outlier count": int(val)} for idx, val in top_outliers.items()]

    strong_pairs = eda["strong_feature_pairs"]
    if strong_pairs:
        pair_rows = [
            {
                "Feature 1": pair["feature_1"],
                "Feature 2": pair["feature_2"],
                "Correlation": format_float(pair["correlation"]),
            }
            for pair in strong_pairs
        ]
        strong_pair_text = markdown_table(pair_rows, ["Feature 1", "Feature 2", "Correlation"])
    else:
        strong_pair_text = "No original feature pair had absolute correlation >= 0.70."

    baseline_rmse = metrics["baseline"]["RMSE"]
    enhanced_rmse = metrics["enhanced"]["RMSE"]
    rmse_delta = baseline_rmse - enhanced_rmse
    r2_delta = metrics["enhanced"]["R2"] - metrics["baseline"]["R2"]
    feature_helped = rmse_delta > 0

    report = f"""# Task 1 - MLP on California Housing

## 1. Dataset Loading

The dataset was loaded from `../california_housing.csv` without modifying the original file. It contains {df.shape[0]} samples, {len(ORIGINAL_FEATURES)} input features, and the target `{TARGET}`.

All columns are numeric and the total number of missing values is {eda["missing_total"]}. The target is measured in hundreds of thousands of dollars. A notable dataset property is top-coding: {eda["target_capped_count"]} rows ({format_float(eda["target_capped_pct"], 2)}%) have `{TARGET}` greater than or equal to 5.0, so the model cannot observe the real values above that cap.

## 2. Exploratory Data Analysis

The EDA artifacts are saved in `figures/` and the tabular summaries are saved as CSV files in this folder.

Main observations:

- `MedInc` has the strongest positive linear relationship with median house value, so income should be one of the most useful predictors.
- `Latitude` and `Longitude` matter because house prices have clear geographic structure, especially around coastal/high-demand regions.
- Several count and ratio variables are highly skewed, especially occupancy, population, and room-related variables. This supports scaling and log/rate feature engineering.
- Outliers are present in occupancy, population, and room variables. MLPs trained with MSE can be sensitive to these points, so validation monitoring and error analysis are important.
- The target cap near 5.0 can make expensive districts hard to fit: multiple true high-end values are collapsed to the same observed target.

Top absolute correlations with `{TARGET}`:

{markdown_table(corr_rows, ["Feature", "Correlation with target"])}

Most skewed variables:

{markdown_table(skew_rows, ["Feature", "Skewness"])}

Largest IQR outlier counts:

{markdown_table(outlier_rows, ["Feature", "IQR outlier count"])}

Strongly correlated original feature pairs:

{strong_pair_text}

Key plots:

- `figures/eda_histograms.png`
- `figures/eda_boxplots.png`
- `figures/eda_correlation_heatmap.png`
- `figures/eda_feature_target_relationships.png`

## 3. Data Preparation

The data was split once at the row-index level and reused for both models:

{markdown_table(split_rows, ["Split", "Rows", "Percent"])}

For each model, `StandardScaler` was fitted only on the training features. The validation and test sets were transformed using the training scaler to avoid data leakage. The scaled arrays were converted to PyTorch tensors through `TensorDataset` and `DataLoader`.

## 4. Baseline MLP Model

The baseline model uses only the original 8 input features. It inherits from `torch.nn.Module` and uses fully connected layers with ReLU activations and dropout:

- Architecture: input -> 128 -> 64 -> 32 -> output
- Loss: Mean Squared Error
- Optimizer: Adam
- Batch size: {baseline_meta["batch_size"]}
- Weight decay: {baseline_meta["weight_decay"]}
- Early stopping patience: {baseline_meta["patience"]}
- Best checkpoint: `{baseline_meta["checkpoint"]}`

Training and validation losses are plotted in `figures/training_curve_baseline.png`.

## 5. Feature Engineering

The enhanced model adds domain-inspired features:

{markdown_table(feature_rows, ["Engineered feature"])}

These features expose ratios, log-compressed skewed variables, nonlinear geographic structure, and approximate distances to Los Angeles and San Francisco. They are intended to help the MLP learn useful tabular relationships without needing to infer every interaction from raw columns alone.

## 6. Enhanced MLP Model

The enhanced model uses the 8 original features plus the engineered features. It also inherits from `torch.nn.Module`:

- Architecture: input -> 160 -> 96 -> 48 -> output
- Loss: Mean Squared Error
- Optimizer: Adam
- Batch size: {enhanced_meta["batch_size"]}
- Weight decay: {enhanced_meta["weight_decay"]}
- Early stopping patience: {enhanced_meta["patience"]}
- Best checkpoint: `{enhanced_meta["checkpoint"]}`

Training and validation losses are plotted in `figures/training_curve_enhanced.png`.

## 7. Model Evaluation and Comparison

Test-set results:

{markdown_table(metric_rows, ["Model", "MAE", "RMSE", "R2", "Best epoch", "Epochs run"])}

The best-performing model by RMSE is **{best_name.title()}**. The enhanced model changed RMSE by {format_float(rmse_delta)} and R2 by {format_float(r2_delta)} relative to the baseline.

Interpretation:

- Lower MAE/RMSE means fewer average and large prediction errors.
- R2 measures the fraction of test-set target variance explained by the model.
- The validation-loss plots in `figures/model_comparison_validation_loss.png` show convergence behavior and whether one model becomes unstable or overfits faster.
- In this run, feature engineering {"improved" if feature_helped else "did not improve"} test RMSE. Even when the gain is modest, engineered features make relevant ratios and nonlinear geography easier for the network to use.

## 8. Error Analysis

The error analysis uses the **{best_name.title()}** model. Plots are saved as:

- `figures/error_actual_vs_predicted.png`
- `figures/error_residual_distribution.png`
- `figures/error_residuals_by_actual.png`

Summary:

- Mean residual: {format_float(error_analysis["mean_residual"])}
- Median absolute error: {format_float(error_analysis["median_absolute_error"])}
- 90th percentile absolute error: {format_float(error_analysis["p90_absolute_error"])}
- 95th percentile absolute error: {format_float(error_analysis["p95_absolute_error"])}
- Test rows at the 5.0 target cap: {error_analysis["capped_target_count_test"]}
- Worst individual predictions are saved in `{error_analysis["worst_prediction_file"]}`.

The hardest cases tend to be high-value districts, capped target values, and points with unusual occupancy, room, or geographic patterns. Large residuals are expected there because the model only sees aggregate district statistics, not direct indicators such as school quality, coastline proximity, local zoning, crime rates, or employment-center access.

## 9. Conclusion and Future Work

This workflow covered loading, EDA, leakage-safe preprocessing, baseline MLP training, manual feature engineering, enhanced MLP training, test evaluation, and error analysis. Standardization was necessary because feature scales differ substantially. Early stopping and checkpointing helped select the best validation model instead of the last epoch.

Can feature engineering still improve deep learning models on structured tabular datasets? Yes. MLPs can learn nonlinear interactions, but tabular datasets are often small enough and structured enough that human-designed ratios, transforms, and geographic features can still improve generalization or training stability. In this experiment, the empirical comparison above is the deciding evidence: the engineered model {"improved RMSE and R2" if feature_helped else "did not beat the baseline on RMSE, but still provided a useful controlled comparison showing that engineered features must be validated rather than assumed beneficial"}.

Future improvements include wider hyperparameter search, batch normalization, Huber loss for outlier robustness, target transformation or explicit handling of the 5.0 cap, geographic clustering, regularization tuning, ensembles, and comparison against strong tabular baselines such as gradient-boosted trees.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    set_reproducibility()
    ensure_dirs()

    df = load_data()
    expected = set(ORIGINAL_FEATURES + [TARGET])
    missing_columns = expected - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing_columns)}")

    eda = compute_eda(df)
    create_eda_plots(df)

    all_indices = np.arange(len(df))
    train_idx, temp_idx = train_test_split(all_indices, test_size=0.30, random_state=SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=SEED)
    split_counts = {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)}

    baseline_split = prepare_split(df, ORIGINAL_FEATURES, train_idx, val_idx, test_idx)

    engineered_df = add_engineered_features(df)
    engineered_features = [col for col in engineered_df.columns if col not in df.columns]
    enhanced_feature_names = ORIGINAL_FEATURES + engineered_features
    enhanced_split = prepare_split(engineered_df, enhanced_feature_names, train_idx, val_idx, test_idx)

    baseline_model, baseline_meta = train_model(
        "baseline",
        baseline_split,
        hidden_dims=(128, 64, 32),
        lr=1e-3,
        batch_size=1024,
        max_epochs=260,
        patience=25,
        weight_decay=1e-4,
        dropout=0.10,
    )
    plot_training_curves(baseline_meta)

    enhanced_model, enhanced_meta = train_model(
        "enhanced",
        enhanced_split,
        hidden_dims=(160, 96, 48),
        lr=1e-3,
        batch_size=1024,
        max_epochs=260,
        patience=25,
        weight_decay=1e-4,
        dropout=0.10,
    )
    plot_training_curves(enhanced_meta)

    metrics = {
        "baseline": evaluate_model(baseline_model, baseline_split),
        "enhanced": evaluate_model(enhanced_model, enhanced_split),
    }
    histories = {"baseline": baseline_meta, "enhanced": enhanced_meta}
    plot_model_comparison(metrics, histories)

    best_name = min(metrics, key=lambda key: metrics[key]["RMSE"])
    if best_name == "baseline":
        error_analysis = create_error_analysis_plots(best_name, baseline_model, baseline_split, test_idx, df)
    else:
        error_analysis = create_error_analysis_plots(best_name, enhanced_model, enhanced_split, test_idx, engineered_df)

    results = {
        "seed": SEED,
        "dataset": str(DATA_PATH.relative_to(ROOT_DIR)),
        "split_counts": split_counts,
        "baseline": baseline_meta,
        "enhanced": enhanced_meta,
        "metrics": metrics,
        "best_model": best_name,
        "error_analysis": error_analysis,
        "engineered_features": engineered_features,
    }

    serializable_results = json.loads(json.dumps(results, default=lambda obj: list(obj) if isinstance(obj, tuple) else str(obj)))
    RESULTS_PATH.write_text(json.dumps(serializable_results, indent=2), encoding="utf-8")
    pd.DataFrame(metrics).T.to_csv(TASK_DIR / "metrics.csv")

    make_report(
        df=df,
        eda=eda,
        split_counts=split_counts,
        baseline_meta=baseline_meta,
        enhanced_meta=enhanced_meta,
        metrics=metrics,
        error_analysis=error_analysis,
        engineered_features=engineered_features,
        best_name=best_name,
    )

    print("Task 1 complete.")
    print(f"Report: {REPORT_PATH}")
    print(f"Metrics: {RESULTS_PATH}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
