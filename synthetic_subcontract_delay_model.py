# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.inspection import permutation_importance
from scipy.stats import randint, uniform



# CONFIG

DATA_PATH = "synthetic_fason_orders.csv"
RANDOM_STATE = 42

RUN_ALL_MODELS = True     # True: ridge + tree + svr + grad_boost
OUTER_FOLDS = 3
INNER_FOLDS = 2
N_ITER = 15
N_JOBS = 1               

OUT_DIR = "outputs_delay_prediction"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
KMEANS_DIR = os.path.join(OUT_DIR, "kmeans")
IMP_DIR = os.path.join(OUT_DIR, "permutation_importance")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(KMEANS_DIR, exist_ok=True)
os.makedirs(IMP_DIR, exist_ok=True)

EPS = 1e-9



# METRICS

def adjusted_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1.0, np.abs(y_true))
    return float(np.mean(np.abs(y_pred - y_true) / denom))

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_pred) / denom)) if False else float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

def adjusted_r2(r2, n, p):
    if n <= p + 1:
        return np.nan
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))

def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))

def compute_metrics(y_true, y_pred, p_features):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    adj = adjusted_r2(r2, n=len(y_true), p=p_features)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "RMSLE": float(rmsle(y_true, y_pred)),
        "R2": float(r2),
        "Adj_R2": float(adj) if not np.isnan(adj) else np.nan,
        "MAPE_adj": float(adjusted_mape(y_true, y_pred)),
        "SMAPE": float(smape(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
    }



# PLOTS

def save_pred_vs_actual(y_true, y_pred, outpath, title):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.35)
    mx = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([0, mx], [0, mx])
    plt.title(title)
    plt.xlabel("Actual Delay_Days_Positive")
    plt.ylabel("Predicted Delay_Days_Positive")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def save_residual_plot(y_true, y_pred, outpath, title):
    resid = np.asarray(y_true) - np.asarray(y_pred)
    plt.figure()
    plt.scatter(y_pred, resid, alpha=0.35)
    plt.axhline(0.0)
    plt.title(title)
    plt.xlabel("Predicted Delay_Days_Positive")
    plt.ylabel("Residual (Actual - Pred)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def save_error_by_decile(y_true, y_pred, outpath, title, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_true - y_pred)

    qs = np.quantile(y_true, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    if len(qs) < 3:
        return

    bin_idx = np.digitize(y_true, qs[1:-1], right=True)

    maes, centers = [], []
    for b in range(len(qs) - 1):
        mask = (bin_idx == b)
        centers.append((qs[b] + qs[b + 1]) / 2)
        maes.append(float(np.mean(abs_err[mask])) if mask.sum() > 0 else np.nan)

    plt.figure()
    plt.plot(centers, maes, marker="o")
    plt.title(title)
    plt.xlabel("Actual-delay bin center (quantiles)")
    plt.ylabel("MAE within bin")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def save_workshop_mae_bar(workshop_ids, y_true, y_pred, outpath, title, min_count=10):
    dfw = pd.DataFrame({
        "Workshop_ID": workshop_ids,
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
    })
    dfw["abs_err"] = np.abs(dfw["y_true"] - dfw["y_pred"])
    grp = dfw.groupby("Workshop_ID")["abs_err"].agg(["mean", "count"]).reset_index()
    grp = grp[grp["count"] >= min_count].sort_values("mean", ascending=False)

    plt.figure()
    plt.bar(grp["Workshop_ID"].astype(str), grp["mean"].astype(float))
    plt.title(title)
    plt.xlabel("Workshop_ID")
    plt.ylabel("MAE (|error|)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()



# LOAD DATA + TARGET

df = pd.read_csv(DATA_PATH, encoding="utf-8")

if "Delay_Days_Positive" not in df.columns:
    df["Delay_Days_Raw"] = df["Actual_LT"] - df["Promised_LT"]
    df["Delay_Days_Positive"] = df["Delay_Days_Raw"].clip(lower=0)

y = df["Delay_Days_Positive"].astype(float).copy()
assert (y >= 0).all()

plt.figure()
plt.hist(df["Delay_Days_Positive"], bins=25)
plt.title("Distribution of Delay_Days_Positive")
plt.xlabel("Delay days (>=0)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "delay_distribution.png"), dpi=300)
plt.close()



# FEATURE ENGINEERING

df["LT_Tightness"] = df["Promised_LT"] / df["Expected_Steps"].replace(0, np.nan)
df["LT_Tightness"] = df["LT_Tightness"].fillna(df["LT_Tightness"].median())

df["Complexity_Score"] = (
    0.3 * df["Revision_Index"]
    + 0.4 * df["Prototype_Flag"]
    + 0.4 * df["Test_Required"]
    + 0.2 * df["Expected_Steps"]
)

df["Workshop_Reliability"] = (
    0.4 * df["Workshop_Skill"]
    + 0.4 * df["Workshop_OTD_Rate"]
    - 0.2 * df["Workshop_Avg_Delay"]
)

df["LT_Slack"] = df["Promised_LT"] - df["Expected_Steps"]
df["Tight_Schedule_Flag"] = (df["LT_Slack"] < 0).astype(int)

df["Congestion_Index"] = (
    0.5 * df["Workload_Level"]
    + 0.3 * df["Workshop_Avg_Delay"]
    + 0.2 * df["Workshop_Delay_Var"]
)

df["Reliability_Ratio"] = df["Workshop_OTD_Rate"] / (1.0 + df["Workshop_Avg_Delay"].clip(lower=0) + EPS)

bins = [0, 2, 4, np.inf]
labels = ["Low", "Medium", "High"]
df["Material_Criticality_Group"] = pd.cut(
    df["Material_Criticality_Num"],
    bins=bins,
    labels=labels,
    include_lowest=True,
)

drop_cols = ["Delay_Days_Positive", "Delay_Days_Raw", "LateFlag", "Actual_LT", "Order_ID"]
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols).copy()
groups_workshop = df["Workshop_ID"].astype(str).copy()



# PREPROCESSING

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = [c for c in X.columns if c not in categorical_cols]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ],
    remainder="drop",
)



# MODELS + SEARCH SPACES

models_and_spaces_all = {
    "ridge": (
        Ridge(random_state=RANDOM_STATE),
        {"model__alpha": uniform(0.05, 200.0)},
    ),
    "decision_tree": (
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        {
            "model__max_depth": [3, 5, 7, 10, None],
            "model__min_samples_split": randint(2, 30),
            "model__min_samples_leaf": randint(1, 15),
        },
    ),
    "svr_rbf": (
        SVR(kernel="rbf"),
        {
            "model__C": uniform(0.5, 30.0),
            "model__gamma": ["scale", "auto"],
            "model__epsilon": uniform(0.0, 0.5),
        },
    ),
    "grad_boost": (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {
            "model__n_estimators": randint(200, 700),
            "model__learning_rate": uniform(0.02, 0.12),
            "model__max_depth": randint(2, 5),
            "model__subsample": uniform(0.7, 0.3),
        },
    ),
}

models_and_spaces = models_and_spaces_all if RUN_ALL_MODELS else {"grad_boost": models_and_spaces_all["grad_boost"]}


# OUTER CV RUNNER

def run_outer_cv(outer_name, outer_splitter, X, y, groups=None):
    all_rows = []
    best_params_rows = []
    oof_store = {m: {"y_true": [], "y_pred": [], "workshop": []} for m in models_and_spaces.keys()}

    if groups is None:
        outer_splits = list(outer_splitter.split(X))
    else:
        outer_splits = list(outer_splitter.split(X, y, groups=groups))

    for model_name, (estimator, space) in models_and_spaces.items():
        print("\n" + "=" * 90)
        print(f"[{outer_name}] MODEL = {model_name}")

        fold_id = 0
        for tr_idx, te_idx in outer_splits:
            fold_id += 1

            X_tr, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
            y_tr, y_te = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()

            pipe = Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])

            rscv = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=space,
                n_iter=N_ITER,
                scoring="neg_root_mean_squared_error",
                cv=INNER_FOLDS,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                refit=True,
                verbose=0,
            )
            rscv.fit(X_tr, y_tr)

            best_pipe = rscv.best_estimator_
            y_pred = best_pipe.predict(X_te)

            # adjusted R2 needs p_features
            X_tr_proc = best_pipe.named_steps["preprocess"].transform(X_tr)
            p_features = int(X_tr_proc.shape[1])

            metrics = compute_metrics(y_te, y_pred, p_features)

            print(
                f"[Fold {fold_id}] RMSE={metrics['RMSE']:.3f} | RMSLE={metrics['RMSLE']:.3f} | "
                f"R2={metrics['R2']:.3f} | Adj_R2={metrics['Adj_R2']:.3f} | best_inner_RMSE={-rscv.best_score_:.3f}"
            )

            all_rows.append({
                "OuterScheme": outer_name,
                "Model": model_name,
                "Fold": fold_id,
                "p_features": p_features,
                **metrics,
            })

            best_params_rows.append({
                "OuterScheme": outer_name,
                "Model": model_name,
                "Fold": fold_id,
                "BestParams": str(rscv.best_params_),
                "BestInnerCV_RMSE": float(-rscv.best_score_),
            })

            oof_store[model_name]["y_true"].extend(list(y_te.values))
            oof_store[model_name]["y_pred"].extend(list(y_pred))
            oof_store[model_name]["workshop"].extend(list(groups_workshop.iloc[te_idx].values))

    results_df = pd.DataFrame(all_rows)
    best_params_df = pd.DataFrame(best_params_rows)

    summary_df = (
        results_df
        .groupby("Model")[["MAE", "RMSE", "RMSLE", "R2", "Adj_R2", "MAPE_adj", "SMAPE", "MedAE"]]
        .agg(["mean", "std"])
    )
    summary_df.columns = [f"{m}_{s}" for (m, s) in summary_df.columns]
    summary_df = summary_df.sort_values(by="RMSE_mean", ascending=True)

    results_df.to_csv(os.path.join(OUT_DIR, f"cv_results_{outer_name}.csv"), index=False, encoding="utf-8")
    best_params_df.to_csv(os.path.join(OUT_DIR, f"best_params_{outer_name}.csv"), index=False, encoding="utf-8")
    summary_df.to_csv(os.path.join(OUT_DIR, f"cv_summary_{outer_name}.csv"), encoding="utf-8")

    print("\n" + "=" * 90)
    print(f"[{outer_name}] SUMMARY:")
    print(summary_df)

    return summary_df, oof_store


def make_oof_plots(oof_store, outer_name):
    for model_name, dd in oof_store.items():
        y_true = np.asarray(dd["y_true"], dtype=float)
        y_pred = np.asarray(dd["y_pred"], dtype=float)
        w_ids = np.asarray(dd["workshop"], dtype=str)

        save_pred_vs_actual(
            y_true, y_pred,
            os.path.join(PLOT_DIR, f"{outer_name}_{model_name}_pred_vs_actual.png"),
            f"{outer_name} | {model_name} | Predicted vs Actual"
        )
        save_residual_plot(
            y_true, y_pred,
            os.path.join(PLOT_DIR, f"{outer_name}_{model_name}_residuals.png"),
            f"{outer_name} | {model_name} | Residuals vs Predicted"
        )
        save_error_by_decile(
            y_true, y_pred,
            os.path.join(PLOT_DIR, f"{outer_name}_{model_name}_error_by_decile.png"),
            f"{outer_name} | {model_name} | MAE by Actual-Delay Quantiles"
        )
        save_workshop_mae_bar(
            w_ids, y_true, y_pred,
            os.path.join(PLOT_DIR, f"{outer_name}_{model_name}_workshop_mae_bar.png"),
            f"{outer_name} | {model_name} | Workshop MAE (OOF)"
        )



# RUN CV (KFold + GroupKFold)

outer_kfold = KFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
outer_groupkfold = GroupKFold(n_splits=OUTER_FOLDS)

summary_k, oof_k = run_outer_cv("outer_kfold", outer_kfold, X, y, groups=None)
summary_g, oof_g = run_outer_cv("outer_groupkfold", outer_groupkfold, X, y, groups=groups_workshop)

make_oof_plots(oof_k, "outer_kfold")
make_oof_plots(oof_g, "outer_groupkfold")



# Fit best model on FULL data (for permutation importance)

def fit_final_best_model(best_model_name: str):
    estimator, space = models_and_spaces_all[best_model_name]
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])

    rscv = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=space,
        n_iter=max(N_ITER, 20),
        scoring="neg_root_mean_squared_error",
        cv=INNER_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        refit=True,
        verbose=0,
    )
    rscv.fit(X, y)
    return rscv.best_estimator_, rscv.best_params_


def run_permutation_importance_on_rawX(best_pipe: Pipeline, tag: str, n_repeats: int = 6):
    """
    SAFE version:
    - permutes ORIGINAL columns in X (before one-hot)
    - so feature list is always X.columns (no length mismatch ever)
    """
    pi = permutation_importance(
        best_pipe,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=1,  # stable
    )

    imp_df = pd.DataFrame({
        "feature": list(X.columns),
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std,
    }).sort_values("importance_mean", ascending=False)

    imp_df.to_csv(os.path.join(IMP_DIR, f"perm_importance_{tag}.csv"), index=False, encoding="utf-8")

    topk = imp_df.head(20).iloc[::-1]
    plt.figure()
    plt.barh(topk["feature"].astype(str), topk["importance_mean"].astype(float))
    plt.title(f"Permutation Importance (Top 20) - {tag}")
    plt.xlabel("Decrease in performance (negRMSE) after permutation")
    plt.tight_layout()
    plt.savefig(os.path.join(IMP_DIR, f"perm_importance_{tag}_top20.png"), dpi=300)
    plt.close()

    return imp_df


best_k_model = str(summary_k.index[0])
best_g_model = str(summary_g.index[0])

print("\nBest model (outer_kfold by RMSE_mean):", best_k_model)
print("Best model (outer_groupkfold by RMSE_mean):", best_g_model)

best_pipe_k, best_params_k = fit_final_best_model(best_k_model)
best_pipe_g, best_params_g = fit_final_best_model(best_g_model)

pd.DataFrame([{"scheme": "outer_kfold", "best_model": best_k_model, "best_params": str(best_params_k)}]).to_csv(
    os.path.join(OUT_DIR, "final_best_params_outer_kfold.csv"), index=False, encoding="utf-8"
)
pd.DataFrame([{"scheme": "outer_groupkfold", "best_model": best_g_model, "best_params": str(best_params_g)}]).to_csv(
    os.path.join(OUT_DIR, "final_best_params_outer_groupkfold.csv"), index=False, encoding="utf-8"
)

_ = run_permutation_importance_on_rawX(best_pipe_k, tag=f"outer_kfold_{best_k_model}", n_repeats=6)
_ = run_permutation_importance_on_rawX(best_pipe_g, tag=f"outer_groupkfold_{best_g_model}", n_repeats=6)



# K-MEANS RISK SEGMENTATION

risk_features = [
    "Complexity_Score", "LT_Tightness", "Congestion_Index", "Workload_Level",
    "Downtime_Risk", "Urgency_Level", "Material_Criticality_Num", "Expected_Steps",
    "Tight_Schedule_Flag", "Quantity", "Revision_Index", "Prototype_Flag", "Test_Required",
]
risk_features = [c for c in risk_features if c in df.columns]

risk_df = df[risk_features].copy().replace([np.inf, -np.inf], np.nan)
risk_df = risk_df.fillna(risk_df.median(numeric_only=True))

scaler = StandardScaler()
risk_X = scaler.fit_transform(risk_df.values)

Ks = list(range(2, 9))
sil_scores = []
for k in Ks:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels_k = km.fit_predict(risk_X)
    sil_scores.append(float(silhouette_score(risk_X, labels_k)))

plt.figure()
plt.plot(Ks, sil_scores, marker="o")
plt.title("K-means Silhouette Score vs K")
plt.xlabel("K (#clusters)")
plt.ylabel("Silhouette score")
plt.tight_layout()
plt.savefig(os.path.join(KMEANS_DIR, "silhouette_vs_k.png"), dpi=300)
plt.close()

best_k = int(Ks[int(np.argmax(sil_scores))])
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
clusters = kmeans.fit_predict(risk_X)

df_km = df.copy()
df_km["RiskCluster"] = clusters

# Ensure LateFlag_calc exists (some datasets may not have it)
df_km["LateFlag_calc"] = (df_km["Delay_Days_Positive"] > 0.5).astype(int)

# --- SAFE aggregation (no lambda inside named aggregation) ---
g = df_km.groupby("RiskCluster")

stats_basic = g["Delay_Days_Positive"].agg(["size", "mean", "median"]).reset_index()
stats_basic = stats_basic.rename(columns={
    "size": "n",
    "mean": "delay_mean",
    "median": "delay_median"
})

# quantile separately (Pandas-safe)
p90 = g["Delay_Days_Positive"].quantile(0.90).reset_index()
p90 = p90.rename(columns={"Delay_Days_Positive": "delay_p90"})

late_rate = g["LateFlag_calc"].mean().reset_index()
late_rate = late_rate.rename(columns={"LateFlag_calc": "late_rate"})

cluster_stats = stats_basic.merge(p90, on="RiskCluster", how="left").merge(late_rate, on="RiskCluster", how="left")
cluster_stats = cluster_stats.sort_values("late_rate", ascending=False)

cluster_stats.to_csv(os.path.join(KMEANS_DIR, "kmeans_cluster_stats.csv"), index=False, encoding="utf-8")

# Boxplot by cluster
plt.figure()
data = [df_km.loc[df_km["RiskCluster"] == c, "Delay_Days_Positive"].values
        for c in sorted(df_km["RiskCluster"].unique())]
labels_box = [str(c) for c in sorted(df_km["RiskCluster"].unique())]
plt.boxplot(data, labels=labels_box, showfliers=False)
plt.title("Delay_Days_Positive by K-means Risk Cluster")
plt.xlabel("RiskCluster")
plt.ylabel("Delay_Days_Positive")
plt.tight_layout()
plt.savefig(os.path.join(KMEANS_DIR, "delay_by_cluster_boxplot.png"), dpi=300)
plt.close()

# Late rate bar
plt.figure()
plt.bar(cluster_stats["RiskCluster"].astype(str), cluster_stats["late_rate"].astype(float))
plt.title("Late Rate by K-means Risk Cluster")
plt.xlabel("RiskCluster")
plt.ylabel("Late rate (Delay > 0.5)")
plt.tight_layout()
plt.savefig(os.path.join(KMEANS_DIR, "late_rate_by_cluster.png"), dpi=300)
plt.close()

print("\n[DONE] Outputs:", OUT_DIR)
print(" - Plots:", PLOT_DIR)
print(" - KMeans:", KMEANS_DIR)
print(" - Permutation importance:", IMP_DIR)
