#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
multidim_instability.py

多維 x 的不穩定度分析腳本：
- 多維連續 x
- 多維離散 x
- 多維連續/離散混合 x
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Literal, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


ArrayLike = np.ndarray


def parse_list_arg(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    text = text.replace("，", ",")
    return [p.strip() for p in text.split(",") if p.strip()]


def load_dataframe(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p, sheet_name=sheet_name)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(p)
    raise ValueError(f"Unsupported file type: {suffix}")


def infer_column_types(df: pd.DataFrame, x_cols: List[str]) -> Tuple[List[str], List[str]]:
    continuous_cols: List[str] = []
    discrete_cols: List[str] = []
    for col in x_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            continuous_cols.append(col)
        else:
            discrete_cols.append(col)
    return continuous_cols, discrete_cols


def build_preprocessor(continuous_cols: List[str], discrete_cols: List[str]) -> ColumnTransformer:
    transformers = []
    if continuous_cols:
        transformers.append((
            "cont",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            continuous_cols,
        ))
    if discrete_cols:
        # sklearn<1.2 uses `sparse`, sklearn>=1.2 uses `sparse_output`
        try:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append((
            "disc",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", onehot),
            ]),
            discrete_cols,
        ))
    if not transformers:
        raise ValueError("No usable X columns were provided.")
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def build_estimator(model_name: str):
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        )
    if model_name == "gbr":
        return GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
    if model_name == "krr":
        return KernelRidge(kernel="rbf", alpha=1e-2, gamma=1.0)
    if model_name == "svr":
        return SVR(kernel="rbf", C=1.0, gamma="scale")
    raise ValueError(f"Unknown model: {model_name}")


def build_model_pipeline(continuous_cols: List[str], discrete_cols: List[str], model_name: str) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(continuous_cols, discrete_cols)),
        ("model", build_estimator(model_name)),
    ])


def normalize_sigma(
    sigma: ArrayLike,
    *,
    method: str = "quantile",
    q_low: float = 0.05,
    q_high: float = 0.95,
    c: Optional[float] = None,
    tol: Optional[float] = None,
    invert: bool = False,
    score_mode: Literal["bounded_01", "spec_ratio"] = "bounded_01",
) -> ArrayLike:
    sigma = np.asarray(sigma, dtype=float)
    eps = 1e-12

    if method == "max":
        denom = (tol + eps) if (tol is not None) else (np.nanmax(sigma) + eps)
        score = sigma / denom
    elif method == "quantile":
        lo = np.nanquantile(sigma, q_low)
        hi = tol if (tol is not None) else np.nanquantile(sigma, q_high)
        score = (sigma - lo) / (hi - lo + eps)
        score = np.clip(score, 0.0, None)
    elif method == "bounded":
        if tol is not None:
            score = sigma / (tol + eps)
        else:
            if c is None:
                c = float(np.nanmedian(sigma))
            score = sigma / (sigma + c + eps)
    else:
        raise ValueError(f"Unknown method: {method}")

    if score_mode == "bounded_01":
        score = np.clip(score, 0.0, 1.0)
    elif score_mode == "spec_ratio":
        score = np.clip(score, 0.0, None)
    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")

    if invert:
        score = 1.0 - score
        if score_mode == "spec_ratio":
            score = np.clip(score, 0.0, None)
    return score


def group_shrink_sigma(sigma_raw: ArrayLike, count: ArrayLike, global_sigma: float, shrink_k: float) -> ArrayLike:
    sigma_raw = np.asarray(sigma_raw, dtype=float)
    count = np.asarray(count, dtype=float)
    w = count / (count + float(shrink_k))
    return w * sigma_raw + (1.0 - w) * float(global_sigma)


def compute_group_metrics(
    df: pd.DataFrame,
    discrete_cols: List[str],
    y_col: str,
    *,
    tol: Optional[float],
    shrink_k: float,
    norm_method: str,
    q_low: float,
    q_high: float,
    c: Optional[float],
) -> pd.DataFrame:
    global_sigma = float(np.std(df[y_col].to_numpy(), ddof=1))
    eps = 1e-12
    g = (
        df.groupby(discrete_cols, dropna=False)[y_col]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mu_hat", "std": "sigma_hat_raw"})
    )
    g["sigma_hat_raw"] = g["sigma_hat_raw"].fillna(0.0)
    g["sigma_hat"] = group_shrink_sigma(
        g["sigma_hat_raw"].to_numpy(),
        g["count"].to_numpy(),
        global_sigma=global_sigma,
        shrink_k=shrink_k,
    )
    g["sigma_vs_global"] = g["sigma_hat"] / (global_sigma + eps)
    g["sigma_vs_tol"] = np.nan if tol is None else g["sigma_hat"] / (tol + eps)
    g["instability_score"] = normalize_sigma(
        g["sigma_hat"].to_numpy(),
        method=norm_method,
        q_low=q_low,
        q_high=q_high,
        c=c,
        tol=tol,
        invert=False,
        score_mode="bounded_01",
    )
    g["stability_score"] = 1.0 - g["instability_score"]
    g["mode"] = "pure_discrete"
    return g


def transform_features_for_support(df_x: pd.DataFrame, continuous_cols: List[str], discrete_cols: List[str]) -> ArrayLike:
    preprocessor = build_preprocessor(continuous_cols, discrete_cols)
    return preprocessor.fit_transform(df_x)


def compute_support_weight_knn(X_support: ArrayLike, *, k: int = 25, support_lambda: float = 2.0) -> ArrayLike:
    X_support = np.asarray(X_support, dtype=float)
    n = X_support.shape[0]
    if n <= 3:
        return np.ones(n, dtype=float)
    k_eff = int(max(2, min(k, n - 1)))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(X_support)
    distances, _ = nn.kneighbors(X_support)
    mean_d = np.mean(distances[:, 1:], axis=1)
    d0 = float(np.median(mean_d))
    eps = 1e-12
    w = d0 / (mean_d + support_lambda * d0 + eps)
    return np.clip(w, 0.0, 1.0)


def fit_model_based_metrics(
    df: pd.DataFrame,
    x_cols: List[str],
    continuous_cols: List[str],
    discrete_cols: List[str],
    y_col: str,
    *,
    model_name: str,
    cv_folds: int,
    tol: Optional[float],
    norm_method: str,
    q_low: float,
    q_high: float,
    c: Optional[float],
    support_shrink: bool,
    support_k: int,
    support_lambda: float,
) -> pd.DataFrame:
    data = df[x_cols + [y_col]].dropna().copy()
    y = data[y_col].to_numpy(dtype=float)
    X_df = data[x_cols].copy()
    global_sigma = float(np.std(y, ddof=1))
    eps = 1e-12

    mu_model = build_model_pipeline(continuous_cols, discrete_cols, model_name)
    var_model = build_model_pipeline(continuous_cols, discrete_cols, model_name)

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    mu_oof = np.zeros_like(y, dtype=float)

    for tr, te in kf.split(X_df):
        m = clone(mu_model)
        m.fit(X_df.iloc[tr], y[tr])
        mu_oof[te] = m.predict(X_df.iloc[te])

    resid2 = (y - mu_oof) ** 2
    var_model.fit(X_df, resid2)
    var_hat = np.maximum(var_model.predict(X_df), 0.0)
    sigma_hat_raw = np.sqrt(var_hat)

    if support_shrink:
        X_support = transform_features_for_support(X_df, continuous_cols, discrete_cols)
        support_w = compute_support_weight_knn(X_support, k=support_k, support_lambda=support_lambda)
        sigma_hat = support_w * sigma_hat_raw + (1.0 - support_w) * global_sigma
    else:
        support_w = np.ones_like(sigma_hat_raw)
        sigma_hat = sigma_hat_raw

    out = data.copy()
    out["mu_hat"] = mu_oof
    out["var_hat"] = var_hat
    out["sigma_hat_raw"] = sigma_hat_raw
    out["support_weight"] = support_w
    out["sigma_hat"] = sigma_hat
    out["sigma_vs_global"] = out["sigma_hat"] / (global_sigma + eps)
    out["sigma_vs_tol"] = np.nan if tol is None else out["sigma_hat"] / (tol + eps)
    out["instability_score"] = normalize_sigma(
        out["sigma_hat"].to_numpy(),
        method=norm_method,
        q_low=q_low,
        q_high=q_high,
        c=c,
        tol=tol,
        invert=False,
        score_mode="bounded_01",
    )
    out["stability_score"] = 1.0 - out["instability_score"]
    out["mode"] = "model_based"
    return out


def make_summary_table(result_df: pd.DataFrame, x_cols: List[str], y_col: str, top_n: int = 20) -> pd.DataFrame:
    cols = x_cols + [y_col, "mu_hat", "sigma_hat_raw", "sigma_hat", "sigma_vs_global", "sigma_vs_tol", "instability_score", "stability_score", "mode"]
    cols = [c for c in cols if c in result_df.columns]
    return result_df[cols].sort_values("sigma_hat", ascending=False).head(top_n)


def save_outputs(result_df: pd.DataFrame, summary_df: pd.DataFrame, out_dir: Path, base_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_excel(out_dir / f"{base_name}.xlsx", index=False)
    result_df.to_csv(out_dir / f"{base_name}.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / f"{base_name}_top_sigma.csv", index=False, encoding="utf-8-sig")


def plot_discrete_top_groups(group_df: pd.DataFrame, discrete_cols: List[str], out_path: Path, top_n: int = 20) -> None:
    plot_df = group_df.sort_values("sigma_hat", ascending=False).head(top_n).copy()
    labels = plot_df[discrete_cols].astype(str).agg(" | ".join, axis=1)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
    x = np.arange(len(plot_df))
    axes[0].bar(x, plot_df["sigma_hat"])
    axes[0].set_ylabel("Layer1\nsigma")
    axes[0].set_title("Top groups by instability")
    axes[1].bar(x, plot_df["sigma_vs_tol"].fillna(0.0))
    axes[1].set_ylabel("Layer2\nsigma/tol")
    axes[2].bar(x, plot_df["instability_score"])
    axes[2].set_ylabel("Layer3\nscore")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=60, ha="right")
    axes[2].set_xlabel("Discrete group combination")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_model_based_top_cases(result_df: pd.DataFrame, x_cols: List[str], out_path: Path, top_n: int = 30) -> None:
    plot_df = result_df.sort_values("sigma_hat", ascending=False).head(top_n).copy()
    labels = plot_df[x_cols].astype(str).agg(" | ".join, axis=1)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 10))
    x = np.arange(len(plot_df))
    axes[0].plot(x, plot_df["sigma_hat_raw"], marker="o", label="sigma_hat_raw")
    axes[0].plot(x, plot_df["sigma_hat"], marker="o", label="sigma_hat")
    axes[0].set_ylabel("Layer1\nsigma")
    axes[0].legend()
    axes[1].plot(x, plot_df["sigma_vs_tol"].fillna(0.0), marker="o")
    axes[1].set_ylabel("Layer2\nsigma/tol")
    axes[2].plot(x, plot_df["instability_score"], marker="o", label="instability")
    axes[2].plot(x, plot_df["stability_score"], marker="o", label="stability")
    axes[2].set_ylabel("Layer3\nscore")
    axes[2].legend()
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=65, ha="right")
    axes[2].set_xlabel("Top cases / local conditions")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_metadata_json(out_dir: Path, meta: dict, base_name: str) -> None:
    (out_dir / f"{base_name}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Multi-dimensional instability analysis")
    parser.add_argument("data_file", type=str)
    parser.add_argument("--sheet_name", type=str, default=None)
    parser.add_argument("-y", "--y_col", type=str, required=True)
    parser.add_argument("-x", "--x_cols", type=str, required=True)
    parser.add_argument("-cc", "--continuous_cols", type=str, default=None)
    parser.add_argument("-dc", "--discrete_cols", type=str, default=None)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "gbr", "krr", "svr"])
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--shrink_k", type=float, default=20.0)
    parser.add_argument("-sush", "--support_shrink", action="store_true")
    parser.add_argument("--support_k", type=int, default=25)
    parser.add_argument("--support_lambda", type=float, default=2.0)
    parser.add_argument("--norm_method", type=str, default="quantile", choices=["max", "quantile", "bounded"])
    parser.add_argument("--q_low", type=float, default=0.05)
    parser.add_argument("--q_high", type=float, default=0.95)
    parser.add_argument("--c", type=float, default=None)
    parser.add_argument("-o", "--out_dir", type=str, default="multidim_instability_out")
    parser.add_argument("--base_name", type=str, default="multidim_instability")
    parser.add_argument("--top_n_plot", type=int, default=20)
    args = parser.parse_args()

    x_cols = parse_list_arg(args.x_cols)
    continuous_cols = parse_list_arg(args.continuous_cols)
    discrete_cols = parse_list_arg(args.discrete_cols)

    df = load_dataframe(args.data_file, args.sheet_name)
    missing = [c for c in ([args.y_col] + x_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if not continuous_cols and not discrete_cols:
        continuous_cols, discrete_cols = infer_column_types(df, x_cols)
    else:
        declared = set(continuous_cols + discrete_cols)
        undeclared = [c for c in x_cols if c not in declared]
        if undeclared:
            auto_cont, auto_disc = infer_column_types(df, undeclared)
            continuous_cols += auto_cont
            discrete_cols += auto_disc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pure_discrete = len(continuous_cols) == 0 and len(discrete_cols) > 0

    if pure_discrete:
        group_df = compute_group_metrics(
            df=df[x_cols + [args.y_col]].dropna().copy(),
            discrete_cols=discrete_cols,
            y_col=args.y_col,
            tol=args.tol,
            shrink_k=args.shrink_k,
            norm_method=args.norm_method,
            q_low=args.q_low,
            q_high=args.q_high,
            c=args.c,
        )
        result_df = df.merge(group_df, on=discrete_cols, how="left")
        summary_df = group_df.sort_values("sigma_hat", ascending=False).head(50)
        save_outputs(result_df, summary_df, out_dir, args.base_name)
        plot_discrete_top_groups(group_df, discrete_cols, out_dir / f"{args.base_name}_plot.png", top_n=args.top_n_plot)
    else:
        result_df = fit_model_based_metrics(
            df=df,
            x_cols=x_cols,
            continuous_cols=continuous_cols,
            discrete_cols=discrete_cols,
            y_col=args.y_col,
            model_name=args.model,
            cv_folds=args.cv_folds,
            tol=args.tol,
            norm_method=args.norm_method,
            q_low=args.q_low,
            q_high=args.q_high,
            c=args.c,
            support_shrink=args.support_shrink,
            support_k=args.support_k,
            support_lambda=args.support_lambda,
        )
        summary_df = make_summary_table(result_df, x_cols, args.y_col, top_n=50)
        save_outputs(result_df, summary_df, out_dir, args.base_name)
        plot_model_based_top_cases(result_df, x_cols, out_dir / f"{args.base_name}_plot.png", top_n=args.top_n_plot)

    meta = {
        "x_cols": x_cols,
        "continuous_cols": continuous_cols,
        "discrete_cols": discrete_cols,
        "y_col": args.y_col,
        "tol": args.tol,
        "model": args.model,
        "mode": "pure_discrete" if pure_discrete else "model_based",
        "support_shrink": bool(args.support_shrink),
        "support_k": args.support_k,
        "support_lambda": args.support_lambda,
        "norm_method": args.norm_method,
        "q_low": args.q_low,
        "q_high": args.q_high,
    }
    save_metadata_json(out_dir, meta, args.base_name)
    print(f"Done. Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
