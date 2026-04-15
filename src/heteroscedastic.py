from typing import Optional, Dict, Literal, List, Any
import numpy as np
import pandas as pd
import argparse
import os
import sys

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KernelDensity


import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from package import LOGger
from package import dataframeprocedure as DFP
from package import visualization3 as vs3
vs3.plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Refactor 規格見同目錄 test_heteroscedastic_開發.iss（工程 sigma / ratio / normalize 分層、雙圖、CLI）
# 連續版「密度感知收縮」見同目錄 規範出有效評估數據單位.iss（約 504–613 行）；實作時可選 sklearn.neighbors（KernelDensity 等）或 scipy.stats.gaussian_kde


def estimate_x_density_weight(
        x: np.ndarray,
        *,
        x_ref: Optional[np.ndarray] = None,
        bandwidth: float = 0.2,
        density_lambda: float = 3.0,
) -> np.ndarray:
    """
    對每個 x（或 grid x）估局部密度 d(x)，再轉成權重 w(x) in [0,1]。
    低密度區 w 小、高密度區 w 大，供 shrink_sigma_by_density 與全域 sigma_global 線性收縮。
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x_ref is None:
        x_ref = x
    x_ref = np.asarray(x_ref, dtype=float).reshape(-1)

    if x.size == 0:
        return np.asarray([], dtype=float)
    if x_ref.size < 2:
        return np.ones_like(x, dtype=float)
    if bandwidth is None or bandwidth <= 0:
        return np.ones_like(x, dtype=float)
    if density_lambda is None or density_lambda <= 0:
        density_lambda = 1e-12

    # 以 x_ref 擬合密度，再在 x 上取 d(x)；最後用 d/(d+lambda*d0) 映射成 [0,1] 權重。
    kde = KernelDensity(kernel="gaussian", bandwidth=float(bandwidth))
    kde.fit(x_ref.reshape(-1, 1))

    log_d_ref = kde.score_samples(x_ref.reshape(-1, 1))
    d_ref = np.exp(log_d_ref)
    d0 = float(np.nanmedian(d_ref))
    if not np.isfinite(d0) or d0 <= 0:
        ok = np.isfinite(d_ref) & (d_ref > 0)
        d0 = float(np.mean(d_ref[ok])) if np.any(ok) else 1.0

    log_d = kde.score_samples(x.reshape(-1, 1))
    d = np.exp(log_d)

    eps = 1e-12
    w = d / (d + float(density_lambda) * (d0 + eps) + eps)
    w = np.clip(w, 0.0, 1.0)
    return w


def shrink_sigma_by_density(
        sigma_local: np.ndarray,
        sigma_global: float,
        weight: np.ndarray,
    ) -> np.ndarray:
    """
    sigma_shrink_density(x) = w(x)*sigma_local(x) + (1-w(x))*sigma_global（與 iss 公式一致）。
    weight 應由 estimate_x_density_weight 產生。
    """
    sl = np.asarray(sigma_local, dtype=float)
    w = np.asarray(weight, dtype=float)
    sg = float(sigma_global)
    return w * sl + (1.0 - w) * sg


def heteroscedastic_metric_layers_spec():
    """
    占位框架：離散與連續路徑統一輸出欄位語意（開發.iss 第 8 節）。
    第一層 sigma_*（有單位）、第二層 sigma_vs_global、第三層 instability_score / stability_score。
    """
    return {
        "layer1_sigma_unit": ["mu_hat", "sigma_hat", "sigma_shrink"],
        "layer2_ratio": ["sigma_vs_global"],
        "layer3_score": ["instability_score", "stability_score"],
    }


def build_three_layer_instability_columns(
        df_metric: pd.DataFrame,
        *,
        sigma_col: str,
        tol: Optional[float],
    ) -> pd.DataFrame:
    eps = 1e-12
    out = pd.DataFrame(index=df_metric.index)
    out["layer1_instability_sigma"] = df_metric[sigma_col]
    if tol is not None:
        out["layer2_instability_sigma_vs_tol"] = df_metric[sigma_col] / (tol + eps)
    else:
        out["layer2_instability_sigma_vs_tol"] = np.nan
    out["layer3_instability_score"] = df_metric["instability_score"]
    out["layer3_stability_score"] = df_metric["stability_score"]
    return out


# python tmp_ysta\test_heteroscedastic.py X:\DataSet\42\202601291436_executing.pkl -x 壓克力鋅鹽_配料重 -y PGA -o tmp_ysta\Proj_68_17

m_output_dir = 'tmp_ysta\output'

def build_model(model_name: str):
    if model_name == 'krr':
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", KernelRidge(kernel="rbf", alpha=1e-2, gamma=1.0)),
        ])

    elif model_name == 'rf':
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=42
        )

    elif model_name == 'gbr':
        return GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

    elif model_name == 'svr':
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=1.0, gamma="scale")),
        ])

    else:
        raise ValueError(f"Unknown model: {model_name}")


def group_stability(df, x_col, y_col, shrink_k=20):
    global_std = df[y_col].std()
    eps = 1e-12

    g = (
        df.groupby(x_col)[y_col]
        .agg(['count', 'mean', 'std'])
        .rename(columns={'std': 'sigma'})
        .reset_index()
    )

    # CV
    g['cv'] = g['sigma'] / g['mean']

    # 標準誤
    g['se_sigma'] = g['sigma'] / np.sqrt(2 * (g['count'] - 1))

    # Shrinkage
    w = g['count'] / (g['count'] + shrink_k)
    g['sigma_shrink'] = w * g['sigma'] + (1 - w) * global_std
    g['se_sigma_shrink'] = w * g['se_sigma']

    # 工程主指標：sigma_shrink（有單位）；跨主題比較：sigma_vs_global（無單位比例）
    g['sigma_vs_global'] = g['sigma_shrink'] / (global_std + eps)
    # Backward-compatible alias（開發.iss 1）：不再拿來放 normalize 分數
    g['instability_ratio'] = g['sigma_vs_global']

    return g


def build_norm_title(
        method: str,
        tol: float,
        q_low: float,
        q_high: float,
        c: float,
        invert: bool,
        display_metric: Optional[str] = None,
    ) -> str:
    # 圖表標題需要呈現 normalize 的刻度設定；先標註 method，再依該 method 是否支援 tol 來附註 tol
    parts = [fr"$norm$=${method}$"]
    if method == "quantile":
        parts.append(fr"$q_{{low}}$={q_low}")
        parts.append(fr"$q_{{high}}$={q_high}")
        if tol is not None:
            parts.append(fr"$\tau$={tol}")
    elif method == "bounded":
        parts.append(fr"$c$={c if c is not None else 'median'}")
        if tol is not None:
            parts.append(fr"$\tau$={tol}")
    elif method == "max":
        if tol is not None:
            parts.append(fr"$\tau$={tol}")
    if tol is not None:
        parts.append(fr"$score=1\leftrightarrow\sigma={tol}$")
    if invert:
        parts.append("invert=True")

    if display_metric:
        parts.append(f"metric={display_metric}")

    return ", ".join(parts)



def normalize_sigma(
        sigma: np.ndarray,
        *,
        method: str = "quantile",   # max | quantile | bounded
        q_low: float = 0.05,
        q_high: float = 0.95,
        c: float = None,
        tol: float = None,
        invert: bool = False,
        score_mode: Literal["bounded_01", "spec_ratio"] = "spec_ratio",
    ):
    """
        將 sigma 映射成分數（可 0~1 或允許 >1）
        當 tol 指定時，score = 1 表示 sigma = tol（y 的震盪為 tol）

        Parameters
        ----------
        score_mode:
            bounded_01：保證 0~1（適合 UI / dashboard）
            spec_ratio：允許 >1（表示超規倍數）
        method:
            "max"       -> sigma / max(sigma) 或 sigma / tol (如果指定 tol)
            "quantile"  -> 分位數縮放 (預設)
            "bounded"   -> sigma / (sigma + c) 或 sigma / tol (如果指定 tol)
        q_low, q_high:
            quantile 模式使用
        c:
            bounded 模式使用（當 tol 未指定時）
        tol:
            製程容差基準，當 score = 1 時對應 sigma = tol
        invert:
            True  -> 1 = 穩定
            False -> 1 = 不穩定
    """

    sigma = np.asarray(sigma)
    eps = 1e-12

    if method == "max":
        if tol is not None:
            # 當指定 tol 時，score = 1 對應 sigma = tol
            score = sigma / (tol + eps)
        else:
            # 原本的行為：score = 1 對應 sigma = max(sigma)
            denom = np.max(sigma) + eps
            score = sigma / denom
        score = np.clip(score, 0, None)  # score_mode=spec_ratio 時允許 > 1

    elif method == "quantile":
        lo = np.quantile(sigma, q_low)
        if tol is not None:
            # 當指定 tol 時，score = 1 對應 sigma = tol
            hi = tol
        else:
            # 原本的行為：score = 1 對應 sigma = q_high 分位數
            hi = np.quantile(sigma, q_high)
        score = (sigma - lo) / (hi - lo + eps)
        score = np.clip(score, 0, None)  # score_mode=spec_ratio 時允許 > 1

    elif method == "bounded":
        if tol is not None:
            # 當指定 tol 時，score = 1 對應 sigma = tol
            score = sigma / (tol + eps)
        else:
            # 原本的行為：score = sigma / (sigma + c)
            if c is None:
                c = np.median(sigma)
            score = sigma / (sigma + c + eps)
        score = np.clip(score, 0, None)  # score_mode=spec_ratio 時允許 > 1

    else:
        raise ValueError(f"Unknown normalize method: {method}")

    if score_mode == "bounded_01":
        score = np.clip(score, 0.0, 1.0)
    elif score_mode == "spec_ratio":
        pass
    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")

    if invert:
        score = 1 - score
        # spec_ratio 時避免因 score>1 造成負值（超規者視為 0）
        if score_mode == "spec_ratio":
            score = np.clip(score, 0.0, None)

    return score


def se_instability_ratio(
        sigma: np.ndarray,
        se_sigma: np.ndarray,
        *,
        method: str = "quantile",
        q_low: float = 0.05,
        q_high: float = 0.95,
        c: float = None,
        tol: float = None,
        invert: bool = False,
    ) -> np.ndarray:
    """
    依 normalize 公式，用 delta method 將 se_sigma 轉換為 se(instability_ratio)
    """
    sigma = np.asarray(sigma)
    se_sigma = np.asarray(se_sigma)
    eps = 1e-12

    if method == "max":
        if tol is not None:
            denom = tol + eps
        else:
            denom = np.max(sigma) + eps
        d_score = 1.0 / denom

    elif method == "quantile":
        lo = np.quantile(sigma, q_low)
        if tol is not None:
            hi = tol
        else:
            hi = np.quantile(sigma, q_high)
        denom = hi - lo + eps
        d_score = 1.0 / denom

    elif method == "bounded":
        if tol is not None:
            denom = tol + eps
            d_score = 1.0 / denom
        else:
            if c is None:
                c = np.median(sigma)
            d_score = c / (sigma + c + eps) ** 2

    else:
        raise ValueError(f"Unknown normalize method: {method}")

    se_ir = np.abs(d_score * se_sigma)
    return se_ir


def se_instability_score(
        sigma: np.ndarray,
        se_sigma: np.ndarray,
        *,
        method: str = "quantile",
        q_low: float = 0.05,
        q_high: float = 0.95,
        c: float = None,
        tol: float = None,
        invert: bool = False,
        score_mode: Literal["bounded_01", "spec_ratio"] = "bounded_01",
    ) -> np.ndarray:
    """
    依 normalize 公式，用 delta method 將 se_sigma 轉換為 se(instability_score)。

    - score_mode='spec_ratio'：對應未截斷的分數（可 >1）
    - score_mode='bounded_01'：0~1 飽和區間近似視為導數=0，因此 se=0
    """
    se_score = se_instability_ratio(
        sigma,
        se_sigma,
        method=method,
        q_low=q_low,
        q_high=q_high,
        c=c,
        tol=tol,
        invert=invert,
    )
    if score_mode == "bounded_01":
        raw_score = normalize_sigma(
            sigma,
            method=method,
            q_low=q_low,
            q_high=q_high,
            c=c,
            tol=tol,
            invert=False,
            score_mode="spec_ratio",
        )
        se_score = np.where((raw_score <= 0.0) | (raw_score >= 1.0), 0.0, se_score)
    elif score_mode == "spec_ratio":
        pass
    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")

    return se_score


def fit_stability_sklearn(
            df,
            x_col,
            y_col,
            *,
            model_name='krr',
            eps=1e-9,
            cv_folds=5,
            random_state=42,
            sigma_normalize_method='quantile',
            sigma_normalize_q_low=0.05,
            sigma_normalize_q_high=0.95,
            sigma_normalize_c=None,
            sigma_normalize_tol=None,
            sigma_normalize_invert=False,
            density_shrink: bool = True,
            density_bandwidth: float = 0.2,
            density_lambda: float = 3.0,
    ) -> pd.DataFrame:
    """
    用連續資料估計「條件平均」與「條件變異」：
      mu(x) = E[y|x]
      var(x) = E[(y-mu(x))^2 | x]
    回傳每筆資料對應的：
    - 第一層（有單位）：mu_hat, sigma_hat
    - 第二層（比例）：sigma_vs_global
    - 第三層（0~1 分數）：instability_score, stability_score

    TODO(規範.iss 2,4,6): sigma_global 一律 np.std(y, ddof=1) 與 density shrink 收斂目標一致；
    保留 sigma_hat，新增 sigma_hat_density_shrink；啟用 density_shrink 時第二、三層改以 shrink 後為主，舊欄位並列供比對。
    """
    data = df[[x_col, y_col]].dropna().copy()
    X = data[[x_col]].to_numpy()
    y = data[y_col].to_numpy()
    global_std = np.std(y, ddof=1)

    # 預設用 KernelRidge 做平滑（非線性、很像核平滑/極限概念）
    mu_model = build_model(model_name)
    var_model = build_model(model_name)

    # 用 out-of-fold 預測避免「同點訓練同點預測」造成變異被低估
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    mu_oof = np.zeros_like(y, dtype=float)

    for train_idx, test_idx in kf.split(X):
        mu_model.fit(X[train_idx], y[train_idx])
        mu_oof[test_idx] = mu_model.predict(X[test_idx])

    resid2 = (y - mu_oof) ** 2

    # 擬合 var(x)
    var_model.fit(X, resid2)
    var_hat = np.maximum(var_model.predict(X), 0.0)
    sigma_hat = np.sqrt(var_hat)

    out = data.copy()
    out["mu_hat"] = mu_oof
    out["var_hat"] = var_hat
    out["sigma_hat"] = sigma_hat
    sigma_hat_density_shrink = out["sigma_hat"].values
    if density_shrink:
        w = estimate_x_density_weight(
            X[:, 0],
            bandwidth=density_bandwidth,
            density_lambda=density_lambda,
        )
        sigma_hat_density_shrink = shrink_sigma_by_density(
            sigma_hat_density_shrink,
            sigma_global=global_std,
            weight=w,
        )
    out["sigma_hat_density_shrink"] = sigma_hat_density_shrink
    out["sigma_vs_global"] = out["sigma_hat_density_shrink"] / (global_std + eps)
    if sigma_normalize_tol is not None:
        out["sigma_vs_tol"] = out["sigma_hat_density_shrink"] / (sigma_normalize_tol + eps)
    else:
        out["sigma_vs_tol"] = np.nan
    out["instability_score"] = normalize_sigma(
        out["sigma_hat_density_shrink"].values,
        method=sigma_normalize_method,
        q_low=sigma_normalize_q_low,
        q_high=sigma_normalize_q_high,
        c=sigma_normalize_c,
        tol=sigma_normalize_tol,
        invert=False,
        score_mode="bounded_01",
    )
    out["stability_score"] = 1.0 - out["instability_score"]
    return out


def plot_data_with_sigma(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        *,
        grid_points: int = 300,
        cv_folds: int = 5,
        output_dir: str = None,
        fig_title: str = None,
        plts: Optional[Dict] = None,
        sigma_normalize_method: str = 'quantile',
        sigma_normalize_q_low: float = 0.05,
        sigma_normalize_q_high: float = 0.95,
        sigma_normalize_c: float = None,
        sigma_normalize_tol: float = None,
        sigma_normalize_invert: bool = False,
        plot_metric: Literal["sigma", "vs_global", "score"] = "sigma",
        density_shrink: bool = True,
        density_bandwidth: float = 0.2,
        density_lambda: float = 3.0,
    ):
    """
    TODO(規範.iss 3): 先算 sigma_local_g，再 sigma_density_shrink_g；三個 subplot 層級一律改畫收縮後曲線。
    """
    data = df[[x_col, y_col]].dropna().copy()
    X = data[[x_col]].to_numpy()
    y = data[y_col].to_numpy()
    global_std = np.std(y, ddof=1)
    eps = 1e-12
    _ = plot_metric  # deprecated: 保留參數僅為 CLI 相容性；連續圖固定輸出三層

    # === 模型 ===
    mu_model = Pipeline([
        ("scaler", StandardScaler()),
        ("krr", KernelRidge(kernel="rbf", alpha=1e-2, gamma=1.0)),
    ])

    var_model = Pipeline([
        ("scaler", StandardScaler()),
        ("krr", KernelRidge(kernel="rbf", alpha=1e-2, gamma=1.0)),
    ])

    # === OOF 平均估計 ===
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    mu_oof = np.zeros_like(y)

    for tr, te in kf.split(X):
        mu_model.fit(X[tr], y[tr])
        mu_oof[te] = mu_model.predict(X[te])

    resid2 = (y - mu_oof) ** 2

    # 用全資料 fit 方便畫曲線
    mu_model.fit(X, y)
    var_model.fit(X, resid2)

    # === 建 grid ===
    x_min, x_max = float(np.min(X)), float(np.max(X))
    Xg = np.linspace(x_min, x_max, grid_points).reshape(-1, 1)

    mu_g = mu_model.predict(Xg)
    var_g = np.maximum(var_model.predict(Xg), 0.0)
    sigma_g = np.sqrt(var_g)
    sigma_local_g = sigma_g
    if density_shrink:
        w_g = estimate_x_density_weight(
            Xg[:, 0],
            x_ref=X[:, 0],
            bandwidth=density_bandwidth,
            density_lambda=density_lambda,
        )
        sigma_density_shrink_g = shrink_sigma_by_density(
            sigma_local=sigma_local_g,
            sigma_global=global_std,
            weight=w_g,
        )
    else:
        sigma_density_shrink_g = sigma_local_g

    sigma_vs_global_g = sigma_density_shrink_g / (global_std + eps)
    tol = sigma_normalize_tol
    has_valid_tol = (tol is not None) and (tol > 0)
    if has_valid_tol:
        sigma_vs_tol_g = sigma_density_shrink_g / (tol + eps)
    instability_score_g = normalize_sigma(
        sigma_density_shrink_g,
        method=sigma_normalize_method,
        q_low=sigma_normalize_q_low,
        q_high=sigma_normalize_q_high,
        c=sigma_normalize_c,
        tol=sigma_normalize_tol,
        invert=False,
        score_mode="bounded_01",
    )
    stability_score_g = 1.0 - instability_score_g

    # === 3-layer subplots ===
    if plts is None:
        plts = {}
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=plts.get("fig_size", (15, 20)))
    ax1, ax2, ax3 = axes

    # 第 1 層：工程主指標（有單位）
    ax1.scatter(
        X[:, 0], y,
        s=12, alpha=0.6,
        color=plts.get("data_color", (0, 0, 1, 0.3)),
        label="Raw data",
    )
    ax1.plot(
        Xg[:, 0], mu_g,
        linewidth=2,
        color=plts.get("mu_color", (0, 0, 1, 0.8)),
        label="mu(x)",
    )
    ax1.set_ylabel(y_col)
    ax1.set_title("Layer 1: Engineering Scale (unit-aware)")
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax1_sigma = ax1.twinx()
    ax1_sigma.plot(
        Xg[:, 0], sigma_density_shrink_g,
        linestyle="--",
        linewidth=1.5,
        color=plts.get("sigma_color", (1, 0, 0)),
        label="sigma(x)",
    )
    ax1_sigma.set_ylabel("Sigma (same unit as y)")

    # 第 2 層：sigma_vs_tol（需要 tol）
    if has_valid_tol:
        ax2.plot(
            Xg[:, 0], sigma_vs_tol_g,
            linewidth=1.5,
            color=plts.get("sigma_color", (1, 0, 0)),
        )
    else:
        msg = "Need --tol to plot Sigma / tol" if tol is None else "Invalid tol (must be > 0) for Sigma / tol"
        ax2.text(0.5, 0.5, msg, transform=ax2.transAxes, ha='center', va='center', fontsize=10)
    ax2.set_ylabel("Sigma / tol")
    ax2.set_title("Layer 2: Sigma / tol")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # 第 3 層：0~1 分數
    score_y = stability_score_g if sigma_normalize_invert else instability_score_g
    score_ylabel = "Stability Score" if sigma_normalize_invert else "Instability Score"
    ax3.plot(
        Xg[:, 0], score_y,
        linewidth=1.5,
        color=plts.get("score_color", (0.2, 0.6, 0.2)),
    )
    ax3.set_ylabel(score_ylabel)
    ax3.set_title("Layer 3: Normalized Score (0-1)")
    ax3.set_xlabel(x_col)
    ax3.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(fig_title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, 'stability.png'))

    return fig, ax1, ax1_sigma


def plot_group_stability(
        g: pd.DataFrame,
        x_col: str,
        *,
        output_dir: str = None,
        fig_title: str = None,
        plts: Optional[Dict] = None,
        sigma_normalize_method: str = 'quantile',
        sigma_normalize_q_low: float = 0.05,
        sigma_normalize_q_high: float = 0.95,
        sigma_normalize_c: float = None,
        sigma_normalize_tol: float = None,
        sigma_normalize_invert: bool = False,
        save_engineering_plot: bool = True,
        save_score_plot: bool = True,
    ):
    if plts is None:
        plts = {}
    _ = (save_engineering_plot, save_score_plot)  # deprecated: 保留參數僅為 CLI 相容性

    x_vals = g[x_col].astype(str)
    x_pos = np.arange(len(g))
    counts = g["count"].values

    y_sigma = g["sigma_shrink"].values
    se_sigma = g["se_sigma_shrink"].values if "se_sigma_shrink" in g.columns else g["se_sigma"].values
    eps = 1e-12
    tol = sigma_normalize_tol
    has_valid_tol = (tol is not None) and (tol > 0)
    if has_valid_tol:
        if "sigma_vs_tol" in g.columns:
            y_vs_tol = g["sigma_vs_tol"].values
        else:
            y_vs_tol = g["sigma_shrink"].values / (tol + eps)

    score_col = "stability_score" if sigma_normalize_invert else "instability_score"
    y_score = g[score_col].values
    se_score = se_instability_score(
        g["sigma_shrink"].values,
        se_sigma,
        method=sigma_normalize_method,
        q_low=sigma_normalize_q_low,
        q_high=sigma_normalize_q_high,
        c=sigma_normalize_c,
        tol=sigma_normalize_tol,
        invert=sigma_normalize_invert,
        score_mode="bounded_01",
    )

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=plts.get("fig_size", (15, 10)))
    ax1, ax2, ax3 = axes

    # 第 1 層：sigma_shrink + error bar
    ax1.bar(
        x_pos,
        y_sigma,
        color=plts.get("bar_color", (0, 0.4, 0.8, 0.6))
    )
    ax1.errorbar(
        x_pos,
        y_sigma,
        yerr=se_sigma,
        fmt='none',
        ecolor='black',
        capsize=4,
        linewidth=1
    )
    sigma_pad = 0.02 * float(np.nanmax(y_sigma)) if np.any(np.isfinite(y_sigma)) else 0.02
    for i, n in enumerate(counts):
        y_pos = float(y_sigma[i]) + float(se_sigma[i]) + sigma_pad
        ax1.text(i, y_pos, f"n={n}", ha='center', va='bottom', fontsize=8)
    ax1.set_ylabel("Sigma_shrink")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # 第 2 層：sigma_vs_tol（需要 tol）
    if has_valid_tol:
        ax2.plot(
            x_pos,
            y_vs_tol,
            marker='o',
            linewidth=1.5,
            color=plts.get("sigma_color", (1, 0, 0))
        )
        vs_pad = 0.02 * float(np.nanmax(y_vs_tol)) if np.any(np.isfinite(y_vs_tol)) else 0.02
        for i, n in enumerate(counts):
            y_pos = float(y_vs_tol[i]) + vs_pad
            ax2.text(i, y_pos, f"n={n}", ha='center', va='bottom', fontsize=8)
    else:
        msg = "Need --tol to plot Sigma / tol" if tol is None else "Invalid tol (must be > 0) for Sigma / tol"
        ax2.text(0.5, 0.5, msg, transform=ax2.transAxes, ha='center', va='center', fontsize=10)
        for i, n in enumerate(counts):
            ax2.text(i, 0.95, f"n={n}", transform=ax2.get_xaxis_transform(), ha='center', va='top', fontsize=8)
    ax2.set_ylabel("Sigma / tol")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # 第 3 層：score + error bar
    ax3.plot(
        x_pos,
        y_score,
        marker='o',
        linewidth=1.5,
        color=plts.get("score_color", (0.2, 0.6, 0.2))
    )
    ax3.errorbar(
        x_pos,
        y_score,
        yerr=se_score,
        fmt='none',
        ecolor='black',
        capsize=4,
        linewidth=1
    )
    score_pad = 0.02
    for i, n in enumerate(counts):
        y_pos = float(y_score[i]) + float(se_score[i]) + score_pad
        ax3.text(i, y_pos, f"n={n}", ha='center', va='bottom', fontsize=8)
    ax3.set_ylabel("Stability Score" if sigma_normalize_invert else "Instability Score")
    ax3.set_xlabel(x_col)
    ax3.grid(True, linestyle="--", alpha=0.3)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_vals)

    fig.suptitle(fig_title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_dir:
        fig.savefig(os.path.join(output_dir, "group_stability.png"))


def main():
    parser = argparse.ArgumentParser(description='異方差性測試')

    parser.add_argument('data_file', type=str, help='資料檔案路徑')
    parser.add_argument('-cf', '--config_file', type=str, help='配置檔案路徑')
    parser.add_argument('-sh', '--sheet_name', type=str, help='Sheet名稱')
    parser.add_argument('-x', '--x_col', type=str, help='x變數名稱')
    parser.add_argument('-y', '--y_col', type=str, help='y變數名稱')
    parser.add_argument('-o', '--output_dir', type=str, help='輸出目錄')
    parser.add_argument(
        '--model',
        type=str,
        default='krr',
        choices=['krr', 'rf', 'gbr', 'svr'],
        help='回歸模型類型'
    )
    parser.add_argument(
        '--norm_method',
        type=str,
        default='quantile',
        choices=['max', 'quantile', 'bounded'],
        help='sigma normalize method'
    )
    parser.add_argument('--q_low', type=float, default=0.05, help='quantile normalize 的下分位數')
    parser.add_argument('--q_high', type=float, default=0.95, help='quantile normalize 的上分位數')
    parser.add_argument(
        '--c',
        type=float,
        default=None,
        help='bounded normalize 的參數 c（當 tol 未指定時使用）'
    )
    parser.add_argument(
        '-t', '--tol',
        type=float,
        default=None,
        help='穩定性門檻對應的 sigma 基準（score=1 時 sigma=tol）；非一般統計意義的 tolerance（見開發.iss 9）'
    )
    parser.add_argument(
        '--plot_metric',
        type=str,
        default='sigma',
        choices=['sigma', 'vs_global', 'score'],
        help='連續圖右軸主指標（開發.iss 9）',
    )
    parser.add_argument(
        '--save_engineering_plot',
        action='store_true',
        help='離散組別圖輸出工程圖（sigma_shrink，見開發.iss 9；兩旗標都不給則預設全輸出）',
    )
    parser.add_argument(
        '--save_score_plot',
        action='store_true',
        help='離散組別圖輸出分數圖（instability/stability_score，見開發.iss 9；兩旗標都不給則預設全輸出）',
    )
    parser.add_argument(
        '--invert',
        action='store_true',
        help='invert scale (1=stable)'
    )
    parser.add_argument(
        '--is_discrete',
        type=lambda x: (str(x).lower() == 'true'),
        default=None,
        help='是否為離散變數 (None=自動判斷)'
    )
    parser.add_argument(
        '--discrete_threshold',
        type=int,
        default=10,
        help='自動判斷離散的閾值（唯一值數量 < 此值則視為離散）'
    )
    parser.add_argument(
        '--shrink_k',
        type=int,
        default=20,
        help='group_stability 的 shrink_k 參數'
    )
    parser.add_argument(
        '--save_raw_with_layers',
        action='store_true',
        help='將原始資料附加三層不穩定度欄位後另存到輸出資料夾',
    )
    parser.add_argument(
        '--raw_layers_fn',
        type=str,
        default='raw_with_instability_layers',
        help='原始資料附加三層不穩定度欄位的輸出檔名（不含副檔名）',
    )
    parser.add_argument(
        '--density_shrink',
        action='store_true',
        default=True,
        help='連續版啟用密度感知 sigma 收縮（規範.iss 5）',
    )
    parser.add_argument(
        '--no_density_shrink',
        action='store_false',
        dest='density_shrink',
        help='連續版停用密度感知 sigma 收縮',
    )
    parser.add_argument(
        '--density_bandwidth',
        type=float,
        default=0.2,
        help='密度權重鄰近範圍（規範.iss 5；交給 estimate_x_density_weight）',
    )
    parser.add_argument(
        '--density_lambda',
        type=float,
        default=3.0,
        help='稀疏區往全域 sigma 收縮強度（規範.iss 5）',
    )


    args = parser.parse_args()
    plts = {
        "fig_size": (15, 30),
        "data_color": (0,0,1,0.3),
        "sigma_color": (1,0,0,0.3)
    }

    df = DFP.import_data(args.data_file, sht=args.sheet_name)
    if not args.output_dir:
        args.output_dir = m_output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # 判斷 x 是否為離散變數
    if args.is_discrete is None:
        # 自動判斷：
        # 1. 如果全部都是非數值，則判斷為離散型
        # 2. 否則，值域唯一量 < threshold 則視為離散
        is_numeric = pd.api.types.is_numeric_dtype(df[args.x_col])
        if not is_numeric:
            is_discrete = True
            print(f"[自動判斷] {args.x_col} 為非數值型，判斷為: 離散")
        else:
            unique_count = df[args.x_col].nunique()
            is_discrete = unique_count < args.discrete_threshold
            print(f"[自動判斷] {args.x_col} 的唯一值數量: {unique_count}, 閾值: {args.discrete_threshold}, 判斷為: {'離散' if is_discrete else '連續'}")
    else:
        is_discrete = args.is_discrete
        print(f"[手動指定] {args.x_col} 判斷為: {'離散' if is_discrete else '連續'}")

    if is_discrete:
        # 離散變數：使用 group_stability
        g = group_stability(
            df,
            args.x_col,
            args.y_col,
            shrink_k=args.shrink_k
        )

        if args.tol is not None:
            g['sigma_vs_tol'] = g['sigma_shrink'] / (args.tol + 1e-12)
         
        # 第三層：instability_score / stability_score（0~1）
        g['instability_score'] = normalize_sigma(
            g['sigma_shrink'].values,
            method=args.norm_method,
            q_low=args.q_low,
            q_high=args.q_high,
            c=args.c,
            tol=args.tol,
            invert=False,
            score_mode="bounded_01",
        )
        g['stability_score'] = 1.0 - g['instability_score']
        
        # 將結果轉換為與連續版本相似的格式
        df_sta = df.merge(
            g[[args.x_col, 'mean', 'sigma', 'sigma_shrink', 'sigma_vs_global', 'instability_score', 'stability_score', 'cv']],
            on=args.x_col,
            how='left'
        )
        df_sta.rename(columns={
            'mean': 'mu_hat',
            'sigma': 'sigma_hat',
        }, inplace=True)
        
        DFP.save(df_sta, exp_fd=args.output_dir, fn='stability', save_types=['xlsx'])

        if args.save_raw_with_layers:
            df_layer = build_three_layer_instability_columns(
                g,
                sigma_col="sigma_shrink",
                tol=args.tol,
            )
            g_layer = pd.concat([g[[args.x_col]], df_layer], axis=1)
            df_raw_with_layers = df.merge(g_layer, on=args.x_col, how='left')
            DFP.save(df_raw_with_layers, exp_fd=args.output_dir, fn=args.raw_layers_fn, save_types=['xlsx'])
        
        # 繪製離散變數圖表
        plot_group_stability(
            g,
            args.x_col,
            output_dir=args.output_dir,
            fig_title=f"Group Stability Analysis - {args.x_col} vs {args.y_col}",
            plts=plts,
            sigma_normalize_method=args.norm_method,
            sigma_normalize_q_low=args.q_low,
            sigma_normalize_q_high=args.q_high,
            sigma_normalize_c=args.c,
            sigma_normalize_tol=args.tol,
            sigma_normalize_invert=args.invert,
            save_engineering_plot=args.save_engineering_plot,
            save_score_plot=args.save_score_plot,
        )
    else:
        # 連續變數：使用 fit_stability_sklearn
        df_sta = fit_stability_sklearn(
            df,
            args.x_col,
            args.y_col,
            model_name=args.model,
            sigma_normalize_method=args.norm_method,
            sigma_normalize_q_low=args.q_low,
            sigma_normalize_q_high=args.q_high,
            sigma_normalize_c=args.c,
            sigma_normalize_tol=args.tol,
            sigma_normalize_invert=args.invert,
            density_shrink=args.density_shrink,
            density_bandwidth=args.density_bandwidth,
            density_lambda=args.density_lambda,
        )

        DFP.save(df_sta, exp_fd=args.output_dir, fn='stability', save_types=['xlsx'])

        if args.save_raw_with_layers:
            df_layer = build_three_layer_instability_columns(
                df_sta,
                sigma_col="sigma_hat_density_shrink",
                tol=args.tol,
            )
            df_raw_with_layers = df.copy()
            for col in df_layer.columns:
                df_raw_with_layers[col] = np.nan
                df_raw_with_layers.loc[df_layer.index, col] = df_layer[col].values
            DFP.save(df_raw_with_layers, exp_fd=args.output_dir, fn=args.raw_layers_fn, save_types=['xlsx'])

        if args.plot_metric == "sigma":
            display_metric = "sigma_hat"
        elif args.plot_metric == "vs_global":
            display_metric = "sigma_vs_global"
        else:
            display_metric = "stability_score" if args.invert else "instability_score"

        norm_title = build_norm_title(
            args.norm_method,
            args.tol,
            args.q_low,
            args.q_high,
            args.c,
            args.invert,
            display_metric=display_metric,
        )
        plot_data_with_sigma(
            df,
            args.x_col,
            args.y_col,
            output_dir=args.output_dir,
            fig_title=f"Stability Analysis - {args.x_col} vs {args.y_col} (global_sigma: {df_sta[args.y_col].std():.4f})\n{norm_title}",
            plts=plts,
            sigma_normalize_method=args.norm_method,
            sigma_normalize_q_low=args.q_low,
            sigma_normalize_q_high=args.q_high,
            sigma_normalize_c=args.c,
            sigma_normalize_tol=args.tol,
            sigma_normalize_invert=args.invert,
            plot_metric=args.plot_metric,
            density_shrink=args.density_shrink,
            density_bandwidth=args.density_bandwidth,
            density_lambda=args.density_lambda,
        )


# --- API 化占位：issues/不穩定度計算功能API化.iss（計算 JSON / 存檔 / 繪圖分離）---


def compute_instability_payload(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        *,
        layers: Optional[List[int]] = None,
        extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    純計算、回傳可經 api_server.safe_serialize 後寫入 JSON 的結構。
    layers: None 表示三層皆要；否則僅處理指定層（1/2/3，語意與現有三層圖一致）。
    """
    if extra is None:
        extra = {}

    def _as_bool(v, default: Optional[bool] = None) -> Optional[bool]:
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, np.integer)) and v in (0, 1):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "t", "1", "yes", "y"):
                return True
            if s in ("false", "f", "0", "no", "n"):
                return False
        return default

    def _as_int(v, default: int) -> int:
        try:
            return int(v)
        except Exception:
            return int(default)

    def _as_float(v, default: Optional[float]) -> Optional[float]:
        if v is None:
            return default
        try:
            vv = float(v)
        except Exception:
            return default
        if not np.isfinite(vv):
            return default
        return vv

    def _resolve_layers(layers_in: Optional[List[int]]) -> List[int]:
        if layers_in is None:
            return [1, 2, 3]
        if not isinstance(layers_in, list):
            raise ValueError("layers must be a list of ints (1/2/3) or None")
        if len(layers_in) == 0:
            return [1, 2, 3]
        out = []
        for v in layers_in:
            try:
                iv = int(v)
            except Exception:
                continue
            if iv in (1, 2, 3) and iv not in out:
                out.append(iv)
        if not out:
            raise ValueError("layers must contain at least one of: 1, 2, 3")
        return out

    def _filter_three_layer_df(df_layer: pd.DataFrame, layers_keep: List[int]) -> pd.DataFrame:
        keep = []
        if 1 in layers_keep:
            keep.append("layer1_instability_sigma")
        if 2 in layers_keep:
            keep.append("layer2_instability_sigma_vs_tol")
        if 3 in layers_keep:
            keep.append("layer3_instability_score")
            keep.append("layer3_stability_score")
        keep = [c for c in keep if c in df_layer.columns]
        return df_layer[keep].copy()

    layers_resolved = _resolve_layers(layers)

    if x_col not in df.columns:
        raise ValueError(f"x_col not found: {x_col}")
    if y_col not in df.columns:
        raise ValueError(f"y_col not found: {y_col}")

    # ---- parameters (from extra, with CLI-aligned defaults) ----
    is_discrete = _as_bool(extra.get("is_discrete"), default=None)
    discrete_threshold = _as_int(extra.get("discrete_threshold"), 10)
    shrink_k = _as_int(extra.get("shrink_k"), 20)

    model_name = str(extra.get("model", "krr"))
    norm_method = str(extra.get("norm_method", "quantile"))
    q_low = _as_float(extra.get("q_low"), 0.05)
    q_high = _as_float(extra.get("q_high"), 0.95)
    c = _as_float(extra.get("c"), None)
    tol = _as_float(extra.get("tol"), None)
    invert = _as_bool(extra.get("invert"), default=False) or False

    density_shrink = _as_bool(extra.get("density_shrink"), default=True)
    density_shrink = True if density_shrink is None else bool(density_shrink)
    density_bandwidth = _as_float(extra.get("density_bandwidth"), 0.2)
    density_lambda = _as_float(extra.get("density_lambda"), 3.0)

    cv_folds = _as_int(extra.get("cv_folds"), 5)
    preview_n = _as_int(extra.get("preview_n"), 5)

    # ---- auto decide discrete/continuous ----
    unique_count = int(df[x_col].nunique(dropna=True))
    if is_discrete is None:
        is_numeric = pd.api.types.is_numeric_dtype(df[x_col])
        if not is_numeric:
            is_discrete = True
        else:
            is_discrete = unique_count < int(discrete_threshold)

    payload: Dict[str, Any] = {
        "implemented": True,
        "x_col": x_col,
        "y_col": y_col,
        "layers_requested": layers,
        "layers_resolved": layers_resolved,
        "is_discrete": bool(is_discrete),
        "unique_count": unique_count,
        "params": {
            "discrete_threshold": int(discrete_threshold),
            "shrink_k": int(shrink_k),
            "model": model_name,
            "norm_method": norm_method,
            "q_low": q_low,
            "q_high": q_high,
            "c": c,
            "tol": tol,
            "invert": bool(invert),
            "cv_folds": int(cv_folds),
            "density_shrink": bool(density_shrink),
            "density_bandwidth": density_bandwidth,
            "density_lambda": density_lambda,
        },
        "notes": [],
    }

    # tol gate for layer2 (sigma/tol)
    if (2 in layers_resolved) and (tol is None):
        payload["notes"].append("Layer2 requested but tol is None; sigma_vs_tol will be NaN / not plottable.")

    if bool(is_discrete):
        g = group_stability(df, x_col, y_col, shrink_k=shrink_k)
        eps = 1e-12
        if tol is not None:
            g["sigma_vs_tol"] = g["sigma_shrink"] / (tol + eps)

        # layer3 score
        if 3 in layers_resolved:
            g["instability_score"] = normalize_sigma(
                g["sigma_shrink"].values,
                method=norm_method,
                q_low=q_low,
                q_high=q_high,
                c=c,
                tol=tol,
                invert=False,
                score_mode="bounded_01",
            )
            g["stability_score"] = 1.0 - g["instability_score"]
        else:
            g["instability_score"] = np.nan
            g["stability_score"] = np.nan

        merge_cols = [
            x_col,
            "mean",
            "sigma",
            "sigma_shrink",
            "sigma_vs_global",
            "instability_score",
            "stability_score",
            "cv",
        ]
        if "sigma_vs_tol" in g.columns:
            merge_cols.append("sigma_vs_tol")
        # 回傳「不含原始其他欄位」的 row-level metrics，避免 API payload 過大
        df_metric = df[[x_col, y_col]].merge(g[merge_cols], on=x_col, how="left")
        df_metric = df_metric.rename(columns={"mean": "mu_hat", "sigma": "sigma_hat"})

        df_layer_g_all = build_three_layer_instability_columns(g, sigma_col="sigma_shrink", tol=tol)
        df_layer_g = _filter_three_layer_df(df_layer_g_all, layers_resolved)
        g_layer = pd.concat([g[[x_col]], df_layer_g], axis=1)

        payload["group_metrics"] = g
        payload["row_metrics"] = df_metric
        payload["layers_table_preview"] = g_layer.head(max(preview_n, 0)).to_dict("records") if preview_n > 0 else []
        payload["layer_columns"] = list(df_layer_g.columns)
        return payload

    # continuous
    df_sta = fit_stability_sklearn(
        df,
        x_col,
        y_col,
        model_name=model_name,
        cv_folds=cv_folds,
        sigma_normalize_method=norm_method,
        sigma_normalize_q_low=q_low,
        sigma_normalize_q_high=q_high,
        sigma_normalize_c=c,
        sigma_normalize_tol=tol,
        sigma_normalize_invert=invert,
        density_shrink=density_shrink,
        density_bandwidth=density_bandwidth,
        density_lambda=density_lambda,
    )
    df_layer_all = build_three_layer_instability_columns(df_sta, sigma_col="sigma_hat_density_shrink", tol=tol)
    df_layer = _filter_three_layer_df(df_layer_all, layers_resolved)
    df_layer_preview = pd.concat([df_sta[[x_col]].copy(), df_layer], axis=1)

    payload["row_metrics"] = df_sta
    payload["layers_table_preview"] = (
        df_layer_preview.head(max(preview_n, 0)).to_dict("records") if preview_n > 0 else []
    )
    payload["layer_columns"] = list(df_layer.columns)
    return payload


def save_instability_to_disk(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        *,
        output_base_dir: str,
        layers: Optional[List[int]] = None,
        extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    與 compute 分離：寫 xlsx/csv 等到 output_base_dir（預設由 API 傳 apiOutput）。
    """
    if extra is None:
        extra = {}
    os.makedirs(output_base_dir, exist_ok=True)

    payload = compute_instability_payload(df, x_col, y_col, layers=layers, extra=extra)
    layers_resolved = payload.get("layers_resolved", [1, 2, 3])
    tol = payload.get("params", {}).get("tol", None)

    def _listdir_set() -> set:
        try:
            return set(os.listdir(output_base_dir))
        except Exception:
            return set()

    def _collect_new_xlsx(prefix: str, before: set) -> List[str]:
        after = _listdir_set()
        new_names = sorted([n for n in (after - before) if isinstance(n, str)])
        out = []
        for name in new_names:
            if not name.lower().endswith(".xlsx"):
                continue
            if not name.startswith(prefix):
                continue
            out.append(os.path.join(output_base_dir, name))
        return out

    # helper: keep only requested layer columns
    def _filter_layer_cols(cols: List[str]) -> List[str]:
        keep = []
        if 1 in layers_resolved:
            keep.append("layer1_instability_sigma")
        if 2 in layers_resolved:
            keep.append("layer2_instability_sigma_vs_tol")
        if 3 in layers_resolved:
            keep.append("layer3_instability_score")
            keep.append("layer3_stability_score")
        return [c for c in cols if c in keep]

    saved_paths: List[str] = []

    # Save the metrics table (stability.xlsx) for traceability
    df_sta = payload.get("row_metrics", None)
    if isinstance(df_sta, pd.DataFrame) and not df_sta.empty:
        # 依 layers 輸出最小必要欄位，避免穩定度資料表過大
        cols: List[str] = []
        if x_col in df_sta.columns:
            cols.append(x_col)
        if y_col in df_sta.columns:
            cols.append(y_col)
        if 1 in layers_resolved:
            cols += [c for c in ["mu_hat", "sigma_hat", "sigma_shrink", "sigma_hat_density_shrink"] if c in df_sta.columns]
        if 2 in layers_resolved:
            cols += [c for c in ["sigma_vs_global", "sigma_vs_tol"] if c in df_sta.columns]
        if 3 in layers_resolved:
            cols += [c for c in ["instability_score", "stability_score"] if c in df_sta.columns]
        cols = list(dict.fromkeys(cols))
        df_sta_to_save = df_sta[cols].copy() if cols else df_sta.copy()

        before = _listdir_set()
        DFP.save(df_sta_to_save, exp_fd=output_base_dir, fn="stability", save_types=["xlsx"])
        expected = os.path.join(output_base_dir, "stability.xlsx")
        saved_paths.extend(_collect_new_xlsx("stability", before) or [expected])

    # Save raw data with requested layer columns
    raw_fn = str(extra.get("raw_layers_fn", "raw_with_instability_layers"))
    if payload.get("is_discrete", False):
        g = payload.get("group_metrics", None)
        if isinstance(g, pd.DataFrame) and not g.empty:
            df_layer_g = build_three_layer_instability_columns(g, sigma_col="sigma_shrink", tol=tol)
            cols_keep = _filter_layer_cols(list(df_layer_g.columns))
            g_layer = pd.concat([g[[x_col]], df_layer_g[cols_keep]], axis=1)
            df_raw_with_layers = df.merge(g_layer, on=x_col, how="left")
            before = _listdir_set()
            DFP.save(df_raw_with_layers, exp_fd=output_base_dir, fn=raw_fn, save_types=["xlsx"])
            expected = os.path.join(output_base_dir, f"{raw_fn}.xlsx")
            saved_paths.extend(_collect_new_xlsx(raw_fn, before) or [expected])
    else:
        if isinstance(df_sta, pd.DataFrame) and not df_sta.empty:
            df_layer = build_three_layer_instability_columns(
                df_sta,
                sigma_col="sigma_hat_density_shrink",
                tol=tol,
            )
            cols_keep = _filter_layer_cols(list(df_layer.columns))
            df_layer = df_layer[cols_keep]
            df_raw_with_layers = df.copy()
            for col in df_layer.columns:
                df_raw_with_layers[col] = np.nan
                df_raw_with_layers.loc[df_layer.index, col] = df_layer[col].values
            before = _listdir_set()
            DFP.save(df_raw_with_layers, exp_fd=output_base_dir, fn=raw_fn, save_types=["xlsx"])
            expected = os.path.join(output_base_dir, f"{raw_fn}.xlsx")
            saved_paths.extend(_collect_new_xlsx(raw_fn, before) or [expected])

    return {
        "implemented": True,
        "output_base_dir": output_base_dir,
        "saved_paths": saved_paths,
        "layers_requested": layers,
        "layers_resolved": layers_resolved,
    }


def plot_instability_figures(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        *,
        output_base_dir: str,
        layers: Optional[List[int]] = None,
        extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    與 compute / save 分離：只負責繪圖輸出檔路徑列表。
    """
    if extra is None:
        extra = {}
    os.makedirs(output_base_dir, exist_ok=True)

    payload = compute_instability_payload(df, x_col, y_col, layers=layers, extra=extra)
    params = payload.get("params", {})

    # Plot defaults (align with CLI)
    plts = {
        "fig_size": (15, 30),
        "data_color": (0, 0, 1, 0.3),
        "sigma_color": (1, 0, 0, 0.3),
    }

    fig_title = extra.get("fig_title", None)
    if not isinstance(fig_title, str) or fig_title.strip() == "":
        if payload.get("is_discrete", False):
            fig_title = f"Group Stability Analysis - {x_col} vs {y_col}"
        else:
            fig_title = f"Stability Analysis - {x_col} vs {y_col}"

    figure_paths: List[str] = []

    if payload.get("is_discrete", False):
        g = payload.get("group_metrics", None)
        if not isinstance(g, pd.DataFrame) or g.empty:
            raise ValueError("Empty group metrics; cannot plot.")
        plot_group_stability(
            g,
            x_col,
            output_dir=output_base_dir,
            fig_title=fig_title,
            plts=plts,
            sigma_normalize_method=str(params.get("norm_method", "quantile")),
            sigma_normalize_q_low=float(params.get("q_low", 0.05)),
            sigma_normalize_q_high=float(params.get("q_high", 0.95)),
            sigma_normalize_c=params.get("c", None),
            sigma_normalize_tol=params.get("tol", None),
            sigma_normalize_invert=bool(params.get("invert", False)),
            save_engineering_plot=True,
            save_score_plot=True,
        )
        figure_paths.append(os.path.join(output_base_dir, "group_stability.png"))
    else:
        norm_title = build_norm_title(
            str(params.get("norm_method", "quantile")),
            params.get("tol", None),
            float(params.get("q_low", 0.05)),
            float(params.get("q_high", 0.95)),
            params.get("c", None),
            bool(params.get("invert", False)),
            display_metric=None,
        )
        data = df[[x_col, y_col]].dropna()
        if not data.empty:
            global_std = float(np.std(data[y_col].to_numpy(dtype=float), ddof=1))
            fig_title = f"{fig_title} (global_sigma: {global_std:.4f})\n{norm_title}"
        plot_data_with_sigma(
            df,
            x_col,
            y_col,
            output_dir=output_base_dir,
            fig_title=fig_title,
            plts=plts,
            sigma_normalize_method=str(params.get("norm_method", "quantile")),
            sigma_normalize_q_low=float(params.get("q_low", 0.05)),
            sigma_normalize_q_high=float(params.get("q_high", 0.95)),
            sigma_normalize_c=params.get("c", None),
            sigma_normalize_tol=params.get("tol", None),
            sigma_normalize_invert=bool(params.get("invert", False)),
            plot_metric="sigma",
            density_shrink=bool(params.get("density_shrink", True)),
            density_bandwidth=float(params.get("density_bandwidth", 0.2)),
            density_lambda=float(params.get("density_lambda", 3.0)),
        )
        figure_paths.append(os.path.join(output_base_dir, "stability.png"))

    return {
        "implemented": True,
        "output_base_dir": output_base_dir,
        "layers_requested": layers,
        "layers_resolved": payload.get("layers_resolved", [1, 2, 3]),
        "figure_paths": figure_paths,
    }


if __name__ == '__main__':
    main()
