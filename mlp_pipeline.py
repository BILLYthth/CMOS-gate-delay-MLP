"""
=============================================================
  mlp_pipeline_v3.py  --  Unified MLP Training Pipeline
  CMOS Gate Propagation Delay Prediction (Tp)
  90nm re-collected datasets (NAND / NOR / EXOR)
=============================================================
Thay doi so voi v2 + mlp_final_old (gop lam 1 file duy nhat):
  - Dataset moi da duoc gop san (1 file/cong: {GATE}_dataset_90nm_multiple.csv)
    -> KHONG can merge single/multiple nua, khong can fix CL=0 (da fix o NOR)
  - R_eq da co san trong file (1 cot duy nhat) -> chi can log10
  - 1 file duy nhat: GridSearchCV (5-fold CV) TIM best hyperparameters,
    ROI TRAIN LAI model tot nhat va xuat toan bo ket qua/plot/model
    (truoc day tach thanh 2 file: mlp_pipeline_v2.py + mlp_final.py)
=============================================================
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# =============================================================
#  CONFIG
# =============================================================
GATES = ["NAND", "NOR", "EXOR"]

DATA_TEMPLATE = "{gate}_dataset_90nm_multiple.csv"

DATA_DIR = Path(".")
OUT_DIR  = Path("./results_v3")

# Cot dinh danh 1 diem do duy nhat (dung de loai trung lap giua cac
# sweep type khac nhau, vi diem baseline lap lai trong moi sweep)
KEY_COLS = ["pin", "VDD", "CL", "Wn", "K", "Temp"]

FEATURE_COLS = [
    "VDD",
    "log_CL",
    "log_Wn",
    "K",
    "Temp",
    "log_Wp",
    "log_Req",
    "pin_enc",
]

# Cong nao khong du R_eq hop le -> dung feature set khong co log_Req
FEATURE_COLS_NO_REQ = [
    "VDD",
    "log_CL",
    "log_Wn",
    "K",
    "Temp",
    "log_Wp",
    "pin_enc",
]

TARGET_COL   = "log_Tp"
RANDOM_STATE = 42
TEST_SIZE    = 0.15
CV_FOLDS     = 5
MLP_MAX_ITER = 5000

PARAM_GRID = {
    "mlp__hidden_layer_sizes": [
        (64, 32),
        (128, 64),
        (128, 64, 32),
        (256, 128, 64),
        (128, 128, 64, 32),
    ],
    "mlp__activation": ["relu", "tanh"],
    "mlp__alpha": [1e-3, 1e-4, 1e-5],
}

FEATURE_LABELS = {
    "VDD":     "VDD (V)",
    "log_CL":  "log10(CL)",
    "log_Wn":  "log10(Wn)",
    "K":       "K = Wp/Wn",
    "Temp":    "Temperature (C)",
    "log_Wp":  "log10(Wp)",
    "log_Req": "log10(R_eq)",
    "pin_enc": "Input Pin (A=1/B=0)",
}
# =============================================================


def load_gate_csv(gate):
    path = DATA_DIR / DATA_TEMPLATE.format(gate=gate)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_dataset(df, gate):
    df = df.copy()
    for col in df.columns:
        if col not in ("gate", "pin", "sweep_type", "source"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["pin"] = df["pin"].astype(str).str.upper().str.strip()

    n0 = len(df)
    df = df[df["Tp"].notna() & (df["Tp"] > 0)]
    df = df[df["CL"] > 0]
    df = df[df["Wn"] >= 1e-9]
    n1 = len(df)
    if n1 < n0:
        print(f"    [fix] loai {n0 - n1} dong khong hop le (Tp<=0 / CL<=0 / Wn qua nho)")

    # Loai trung lap: diem baseline lap lai giua cac sweep type
    n2 = len(df)
    df = df.drop_duplicates(subset=KEY_COLS, keep="first").reset_index(drop=True)
    if len(df) < n2:
        print(f"    [fix] loai {n2 - len(df)} dong trung lap (baseline points giua cac sweep)")

    return df


def feature_engineering(df):
    df = df.copy()
    df["Wp"] = df["Wn"] * df["K"]
    for col, new in [("CL", "log_CL"), ("Wn", "log_Wn"), ("Wp", "log_Wp")]:
        df[new] = np.log10(df[col].clip(lower=1e-20))
    if "R_eq" in df.columns:
        df["log_Req"] = np.log10(df["R_eq"].clip(lower=1e-3))
    df["pin_enc"] = (df["pin"].astype(str).str.upper() == "A").astype(int)
    df["log_Tp"] = np.log10(df["Tp"].clip(lower=1e-20))
    return df


def plot_actual_vs_predicted(y_true, y_pred, gate, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true * 1e9, y_pred * 1e9, alpha=0.6, s=30)
    lim = max(y_true.max(), y_pred.max()) * 1.05 * 1e9
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Ideal")
    ax.set_xlabel("Actual Tp (ns)")
    ax.set_ylabel("Predicted Tp (ns)")
    ax.set_title(f"{gate} - Actual vs Predicted Tp (Test)")
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / f"{gate}_actual_vs_predicted.png", dpi=150)
    plt.close()


def plot_error_distribution(pct_err, gate, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pct_err, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Error (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"{gate} - Test Set % Error Distribution")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / f"{gate}_error_distribution.png", dpi=150)
    plt.close()


def plot_feature_importance(model, X_test, y_test, feat_cols, gate, out_dir):
    result = permutation_importance(
        model, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1
    )
    order = result.importances_mean.argsort()
    labels = [FEATURE_LABELS.get(feat_cols[i], feat_cols[i]) for i in order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(labels, result.importances_mean[order],
            xerr=result.importances_std[order], color="seagreen")
    ax.set_xlabel("Permutation Importance (drop in R2)")
    ax.set_title(f"{gate} - Feature Importance")
    plt.tight_layout()
    plt.savefig(out_dir / f"{gate}_feature_importance.png", dpi=150)
    plt.close()


def run_gate(gate):
    print(f"\n{'='*60}")
    print(f"  GATE: {gate}")
    print(f"{'='*60}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_gate_csv(gate)
    if df_raw is None:
        print(f"  [LOI] Khong tim thay {DATA_TEMPLATE.format(gate=gate)}, bo qua.")
        return None
    print(f"  Loaded: {len(df_raw)} dong")

    print(f"\n  [1] Lam sach du lieu...")
    df = clean_dataset(df_raw, gate)
    print(f"    Con lai: {len(df)} dong")

    print(f"\n  [2] Feature engineering...")
    df = feature_engineering(df)
    feat_cols = [f for f in FEATURE_COLS if f in df.columns]
    missing = [f for f in FEATURE_COLS if f not in df.columns]
    if missing:
        print(f"    [CANH BAO] Thieu feature: {missing} -> bo qua")

    df_clean = df[feat_cols + [TARGET_COL]].dropna()
    n_dropped = len(df) - len(df_clean)

    if n_dropped > len(df) * 0.20 and "log_Req" in feat_cols:
        print(f"    [CANH BAO] Mat {n_dropped} dong ({n_dropped/len(df)*100:.0f}%) "
              f"do log_Req NaN -> chuyen sang feature set khong co R_eq")
        feat_cols = [f for f in FEATURE_COLS_NO_REQ if f in df.columns]
        df_clean = df[feat_cols + [TARGET_COL]].dropna()
        n_dropped = len(df) - len(df_clean)
        print(f"    [fix] Feature set moi: {feat_cols}")

    if n_dropped > 0:
        print(f"    [fix] loai {n_dropped} dong co NaN trong features")

    X = df_clean[feat_cols].values
    y = df_clean[TARGET_COL].values
    print(f"    Features ({len(feat_cols)}): {feat_cols}")
    print(f"    Tp range: {10**y.min():.3e} -> {10**y.max():.3e} s")
    print(f"    Samples : {len(X)}")

    print(f"\n  [3] Tach test set {int(TEST_SIZE*100)}%...")
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"    Train+Val={len(X_tv)}, Test={len(X_test)}")

    n_combos = len(PARAM_GRID["mlp__hidden_layer_sizes"]) * \
        len(PARAM_GRID["mlp__activation"]) * len(PARAM_GRID["mlp__alpha"])
    print(f"\n  [4] GridSearchCV ({CV_FOLDS}-fold CV, {n_combos} combinations)...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            max_iter=MLP_MAX_ITER,
            early_stopping=True,
            n_iter_no_change=50,
            random_state=RANDOM_STATE,
            verbose=False,
        ))
    ])
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, PARAM_GRID, cv=cv,
                       scoring="r2", n_jobs=-1, verbose=0, refit=True)
    gs.fit(X_tv, y_tv)

    print(f"    Best params : {gs.best_params_}")
    print(f"    Best CV R2  : {gs.best_score_:.4f}")

    # gs.best_estimator_ da duoc refit tren toan bo X_tv/y_tv (refit=True)
    # -> day chinh la model cuoi cung, khong can train lai lan nua
    best_model = gs.best_estimator_

    print(f"\n  [5] Evaluate...")

    def eval_set(X_in, y_true_log, label):
        y_pred_log = best_model.predict(X_in)
        y_true = 10 ** y_true_log
        y_pred = 10 ** y_pred_log
        pct_err = np.abs((y_true - y_pred) / y_true) * 100
        mape = float(np.mean(pct_err))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true_log, y_pred_log))
        print(f"    [{label:8s}] MAPE={mape:6.2f}%  MAE={mae:.4e}s  R2(log)={r2:.4f}")
        return {"label": label, "MAPE_%": mape, "MAE_s": mae, "R2_log": r2}, y_pred, pct_err

    m_tv, _, _ = eval_set(X_tv, y_tv, "Train+Val")
    m_test, y_pred_test, pct_err_test = eval_set(X_test, y_test, "Test")

    print(f"\n  [6] Ve plot...")
    plot_actual_vs_predicted(10**y_test, y_pred_test, gate, OUT_DIR)
    plot_error_distribution(pct_err_test, gate, OUT_DIR)
    plot_feature_importance(best_model, X_test, y_test, feat_cols, gate, OUT_DIR)

    # GridSearch results table
    gs_df = pd.DataFrame(gs.cv_results_)
    gs_df = gs_df[["param_mlp__hidden_layer_sizes", "param_mlp__activation",
                    "param_mlp__alpha", "mean_test_score", "std_test_score", "rank_test_score"]]
    gs_df = gs_df.sort_values("rank_test_score")
    gs_df.to_csv(OUT_DIR / f"{gate}_gridsearch_results.csv", index=False)
    print(f"\n    Top 5 combinations:")
    print(gs_df.head(5).to_string(index=False))

    pd.DataFrame([m_tv, m_test]).to_csv(OUT_DIR / f"{gate}_metrics_final.csv", index=False)

    pd.DataFrame({
        "Actual_Tp_s": 10**y_test,
        "Predicted_Tp_s": y_pred_test,
        "AbsError_ps": np.abs(10**y_test - y_pred_test) * 1e12,
        "Error_%": pct_err_test,
    }).to_csv(OUT_DIR / f"{gate}_test_predictions_final.csv", index=False)

    # Luu model cuoi cung (pipeline: scaler + mlp) kem feature list va best params
    joblib.dump({
        "model": best_model,
        "feature_cols": feat_cols,
        "best_params": gs.best_params_,
    }, OUT_DIR / f"{gate}_final_model.pkl")
    print(f"\n    Model da luu: {gate}_final_model.pkl")

    return {**m_test, "best_params": gs.best_params_}


def main():
    print("CMOS Gate Tp Prediction -- Unified MLP Pipeline v3")
    print(f"Data dir : {DATA_DIR.resolve()}")
    print(f"Output   : {OUT_DIR.resolve()}")

    summary = []
    for gate in GATES:
        result = run_gate(gate)
        if result:
            summary.append({"Gate": gate, **result})

    if summary:
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        df_sum = pd.DataFrame(summary)
        print(df_sum.to_string(index=False))
        df_sum.to_csv(OUT_DIR / "summary_all_gates_v3.csv", index=False)
        print(f"\nTat ca ket qua luu tai: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
