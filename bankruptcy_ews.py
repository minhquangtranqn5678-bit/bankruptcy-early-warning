import sys, subprocess
def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
    _pip("kagglehub", "imbalanced-learn", "xgboost", "lightgbm", "shap")
except ImportError:
    IN_COLAB = False

# %%
import os, json, time, shutil, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, cross_val_predict, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (recall_score, precision_score, f1_score, fbeta_score,
                             roc_auc_score, average_precision_score, accuracy_score,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             ConfusionMatrixDisplay, make_scorer)

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

# Thư viện tuỳ chọn: nếu thiếu, notebook vẫn chạy với các phần còn lại
try:
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    from sklearn.pipeline import Pipeline
    HAS_IMBLEARN = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

OUT = "results"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    if "ipykernel" in sys.modules:   # trong Jupyter/Colab: hiển thị hình
        plt.show()
    else:                            # chạy file .py (PyCharm): chỉ lưu hình, không mở cửa sổ
        plt.close("all")
        print("  -> đã lưu", os.path.join(FIG, name))
print({"imblearn": HAS_IMBLEARN, "xgboost": HAS_XGB, "lightgbm": HAS_LGBM, "shap": HAS_SHAP})

DATA_PATH = "data.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
TARGET = "Bankrupt?"
print("Kích thước:", df.shape)
print("Số giá trị thiếu:", int(df.isna().sum().sum()))
print("Số dòng trùng lặp:", int(df.duplicated().sum()))
df.head()

counts = df[TARGET].value_counts().sort_index()
rate = counts.get(1, 0) / len(df)
print(f"Không phá sản: {counts.get(0,0)} | Phá sản: {counts.get(1,0)} | Tỷ lệ phá sản: {rate:.2%}")

fig, ax = plt.subplots(figsize=(5, 3.5))
bars = ax.bar(["Không phá sản (0)", "Phá sản (1)"], counts.values, color=["#4C72B0", "#C44E52"])
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v/len(df):.1%})", ha="center", va="bottom")
ax.set_ylabel("Số doanh nghiệp")
ax.set_title("Hình 2. Phân bố nhãn: dữ liệu mất cân bằng nặng")
ax.set_ylim(0, counts.max() * 1.18)
savefig("fig02_label_distribution.png")

# %%
X_all = df.drop(columns=[TARGET])
y_all = df[TARGET].astype(int)

# Tương quan (Pearson) của từng chỉ số với biến mục tiêu
corr_target = X_all.corrwith(y_all).dropna().sort_values(key=np.abs, ascending=False)
top_corr = corr_target.head(15)
fig, ax = plt.subplots(figsize=(7, 5))
top_corr[::-1].plot.barh(ax=ax, color=["#C44E52" if v > 0 else "#4C72B0" for v in top_corr[::-1]])
ax.set_xlabel("Hệ số tương quan với Bankrupt?")
ax.set_title("Hình 3. 15 chỉ số tương quan mạnh nhất với phá sản")
savefig("fig03_top_correlations.png")
top_corr.to_csv(os.path.join(OUT, "top15_correlation_with_target.csv"), header=["corr"])

# %%
top20 = corr_target.head(20).index
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(X_all[top20].corr(), cmap="RdBu_r", center=0, ax=ax,
            cbar_kws={"shrink": .7}, xticklabels=True, yticklabels=True)
ax.set_title("Hình 4. Ma trận tương quan của 20 chỉ số quan trọng nhất")
ax.tick_params(labelsize=7)
savefig("fig04_corr_heatmap.png")

# %%
# So sánh phân bố một số chỉ số giữa 2 nhóm
show = list(corr_target.head(6).index)
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for ax, col in zip(axes.ravel(), show):
    sns.boxplot(data=df, x=TARGET, y=col, ax=ax, showfliers=False, palette=["#4C72B0", "#C44E52"])
    ax.set_title(col[:45], fontsize=9); ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticklabels(["Không PS", "Phá sản"])
fig.suptitle("Hình 5. Phân bố 6 chỉ số then chốt theo nhóm doanh nghiệp")
savefig("fig05_boxplots.png")

# %% [markdown]
# ## 3. Tiền xử lý và chia dữ liệu
# - Loại cột hằng số (ví dụ `Net Income Flag`)
# - Chia train/test 80/20, **stratified** để giữ tỷ lệ phá sản
# - Loại các chỉ số tương quan > 0,9 với chỉ số khác (**chỉ tính trên tập train** để tránh rò rỉ dữ liệu)

# %%
const_cols = [c for c in X_all.columns if X_all[c].nunique() <= 1]
X_all = X_all.drop(columns=const_cols)
print("Cột hằng số bị loại:", const_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED)

corr_m = X_train.corr().abs()
upper = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
high_corr = [c for c in upper.columns if (upper[c] > 0.9).any()]
X_train = X_train.drop(columns=high_corr)
X_test = X_test.drop(columns=high_corr)
FEATURES = list(X_train.columns)

prep_info = {
    "n_rows": int(len(df)), "n_features_raw": int(df.shape[1] - 1),
    "n_bankrupt": int(counts.get(1, 0)), "bankrupt_rate": float(rate),
    "constant_cols_removed": const_cols, "high_corr_removed": high_corr,
    "n_features_final": len(FEATURES),
    "train_size": int(len(X_train)), "test_size": int(len(X_test)),
    "train_bankrupt": int(y_train.sum()), "test_bankrupt": int(y_test.sum()),
}
print(json.dumps({k: v for k, v in prep_info.items() if k != "high_corr_removed"}, ensure_ascii=False, indent=1))
print(f"Loại {len(high_corr)} chỉ số tương quan cao; còn lại {len(FEATURES)} chỉ số.")

# %% [markdown]
# ## 4. So sánh 12 cấu hình: 4 mô hình × 3 cách xử lý mất cân bằng
# - **none:** không xử lý
# - **weight:** trọng số lớp (`class_weight` / `scale_pos_weight` / `is_unbalance`)
# - **smote:** SMOTE nằm **trong pipeline**, nên chỉ sinh mẫu trên phần train của từng fold
#
# Đánh giá bằng Stratified 5-fold CV trên tập train.

# %%
neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
SPW = neg / pos

def make_model(name, strategy):
    w = strategy == "weight"
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=3000, class_weight="balanced" if w else None, random_state=SEED)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                      class_weight="balanced_subsample" if w else None, random_state=SEED)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1,
                             scale_pos_weight=SPW if w else 1.0, random_state=SEED)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, is_unbalance=w,
                              verbose=-1, random_state=SEED)
    raise ValueError(name)

def make_pipe(name, strategy, model=None):
    steps = []
    if name == "LogisticRegression":
        steps.append(("scaler", StandardScaler()))
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("model", model if model is not None else make_model(name, strategy)))
    return Pipeline(steps)

MODELS = ["LogisticRegression", "RandomForest"] + (["XGBoost"] if HAS_XGB else []) + (["LightGBM"] if HAS_LGBM else [])
STRATS = ["none", "weight"] + (["smote"] if HAS_IMBLEARN else [])

f2 = make_scorer(fbeta_score, beta=2, zero_division=0)
SCORING = {"recall": "recall", "precision": make_scorer(precision_score, zero_division=0),
           "f1": "f1", "f2": f2, "roc_auc": "roc_auc", "pr_auc": "average_precision"}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

rows = []
for m in MODELS:
    for s in STRATS:
        t0 = time.time()
        res = cross_validate(make_pipe(m, s), X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
        row = {"model": m, "strategy": s}
        for k in SCORING:
            row[f"{k}_mean"] = res[f"test_{k}"].mean()
            row[f"{k}_std"] = res[f"test_{k}"].std()
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"{m:18s} {s:7s} recall={row['recall_mean']:.3f} f2={row['f2_mean']:.3f} "
              f"roc_auc={row['roc_auc_mean']:.3f} pr_auc={row['pr_auc_mean']:.3f} ({row['seconds']}s)")

cv_df = pd.DataFrame(rows)
cv_df.to_csv(os.path.join(OUT, "table_cv_12_configs.csv"), index=False)
cv_df.sort_values("pr_auc_mean", ascending=False).round(3)

# %%
# Biểu đồ so sánh Recall và PR-AUC của 12 cấu hình
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric, title in [(axes[0], "recall_mean", "Recall (lớp phá sản)"), (axes[1], "pr_auc_mean", "PR-AUC")]:
    p = cv_df.pivot(index="model", columns="strategy", values=metric).reindex(MODELS)[STRATS]
    p.plot.bar(ax=ax, rot=0, color=["#BBBBBB", "#4C72B0", "#C44E52"][:len(STRATS)])
    ax.set_title(title); ax.set_xlabel(""); ax.set_ylim(0, 1); ax.legend(title="Xử lý mất cân bằng")
fig.suptitle("Hình 6. Tác động của cách xử lý mất cân bằng (5-fold CV trên tập train)")
savefig("fig06_imbalance_strategies.png")

# %% [markdown]
# ## 5. Chọn và tinh chỉnh mô hình tốt nhất
# Chọn cấu hình có **PR-AUC** trung bình cao nhất (phù hợp dữ liệu mất cân bằng), sau đó tinh chỉnh siêu tham số
# bằng RandomizedSearchCV (tối ưu PR-AUC).

# %%
best_row = cv_df.sort_values("pr_auc_mean", ascending=False).iloc[0]
BEST_MODEL, BEST_STRAT = best_row["model"], best_row["strategy"]
print("Cấu hình tốt nhất theo CV:", BEST_MODEL, "+", BEST_STRAT)

GRIDS = {
    "LogisticRegression": {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
    "RandomForest": {"model__n_estimators": [200, 300, 400], "model__max_depth": [None, 8, 12, 20],
                     "model__min_samples_leaf": [1, 2, 4, 8], "model__max_features": ["sqrt", 0.3, 0.5]},
    "XGBoost": {"model__n_estimators": [200, 400, 600], "model__max_depth": [3, 4, 5, 6],
                "model__learning_rate": [0.02, 0.05, 0.1], "model__subsample": [0.7, 0.8, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0], "model__min_child_weight": [1, 3, 5]},
    "LightGBM": {"model__n_estimators": [200, 400, 600], "model__num_leaves": [7, 15, 31],
                 "model__learning_rate": [0.02, 0.05, 0.1], "model__min_child_samples": [10, 20, 40],
                 "model__colsample_bytree": [0.6, 0.8, 1.0]},
}
n_iter = min(15,int(np.prod([len(v) for v in GRIDS[BEST_MODEL].values()])))
search = RandomizedSearchCV(make_pipe(BEST_MODEL, BEST_STRAT), GRIDS[BEST_MODEL], n_iter=n_iter,
                            scoring="average_precision", cv=cv, random_state=SEED, n_jobs=1)
search.fit(X_train, y_train)
best_pipe = search.best_estimator_
print("Tham số tốt nhất:", search.best_params_)
print(f"PR-AUC CV trước tinh chỉnh: {best_row['pr_auc_mean']:.3f} | sau: {search.best_score_:.3f}")

# %% [markdown]
# ### Chọn ngưỡng cảnh báo
# Ngưỡng mặc định 0,5 thường bỏ sót nhiều doanh nghiệp phá sản. Ngưỡng được chọn để **tối đa F2-score**
# (ưu tiên Recall) dựa trên dự đoán *out-of-fold* của tập train — **không dùng tập test**.

# %%
oof = cross_val_predict(best_pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
ths = np.round(np.arange(0.01, 0.96, 0.01), 2)
f2s = [fbeta_score(y_train, (oof >= t).astype(int), beta=2, zero_division=0) for t in ths]
recs = [recall_score(y_train, (oof >= t).astype(int), zero_division=0) for t in ths]
precs = [precision_score(y_train, (oof >= t).astype(int), zero_division=0) for t in ths]
THRESH = float(ths[int(np.argmax(f2s))])
print("Ngưỡng cảnh báo được chọn:", THRESH)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ths, recs, label="Recall"); ax.plot(ths, precs, label="Precision"); ax.plot(ths, f2s, label="F2", lw=2)
ax.axvline(THRESH, ls="--", c="k"); ax.text(THRESH + 0.01, 0.95, f"ngưỡng = {THRESH}", va="top")
ax.set_xlabel("Ngưỡng xác suất"); ax.set_ylabel("Giá trị"); ax.legend()
ax.set_title("Hình 7. Recall / Precision / F2 theo ngưỡng (out-of-fold, tập train)")
savefig("fig07_threshold.png")

# %% [markdown]
# ## 6. Đánh giá trên tập test (20%, chưa từng dùng khi huấn luyện)
# Mỗi mô hình lấy cách xử lý mất cân bằng tốt nhất của nó (theo CV), huấn luyện lại trên toàn bộ train và đánh giá trên test.

# %%
def metrics_at(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": t, "accuracy": accuracy_score(y, pred),
            "recall": recall_score(y, pred, zero_division=0), "precision": precision_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "f2": fbeta_score(y, pred, beta=2, zero_division=0),
            "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn)}

test_rows, probas = [], {}
# Mô hình "ngây thơ": đoán tất cả không phá sản
naive = metrics_at(y_test.values, np.zeros(len(y_test)), 0.5)
naive.update({"model": "Baseline (đoán tất cả không phá sản)", "strategy": "-"})
naive["roc_auc"] = naive["pr_auc"] = np.nan
test_rows.append(naive)

for m in MODELS:
    s = cv_df[cv_df.model == m].sort_values("pr_auc_mean", ascending=False).iloc[0]["strategy"]
    pipe = best_pipe if m == BEST_MODEL else make_pipe(m, s).fit(X_train, y_train)
    label = f"{m} (đã tinh chỉnh)" if m == BEST_MODEL else m
    p = pipe.predict_proba(X_test)[:, 1]
    probas[label] = p
    r = metrics_at(y_test.values, p, 0.5); r.update({"model": label, "strategy": s}); test_rows.append(r)
    if m == BEST_MODEL:
        r2 = metrics_at(y_test.values, p, THRESH)
        r2.update({"model": f"{label} @ ngưỡng {THRESH}", "strategy": s}); test_rows.append(r2)

test_df = pd.DataFrame(test_rows)[["model", "strategy", "threshold", "accuracy", "recall", "precision",
                                   "f1", "f2", "roc_auc", "pr_auc", "TP", "FN", "FP", "TN"]]
test_df.to_csv(os.path.join(OUT, "table_test_results.csv"), index=False)
test_df.round(3)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for label, p in probas.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    axes[0].plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_test, p):.3f})")
    pr, rc, _ = precision_recall_curve(y_test, p)
    axes[1].plot(rc, pr, label=f"{label} (AP={average_precision_score(y_test, p):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1); axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("Đường cong ROC")
axes[1].axhline(y_test.mean(), ls="--", c="k", lw=1, label=f"Ngẫu nhiên ({y_test.mean():.3f})")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Đường cong Precision–Recall")
for ax in axes: ax.legend(fontsize=8)
fig.suptitle("Hình 8. So sánh các mô hình trên tập test")
savefig("fig08_roc_pr_curves.png")

# %%
best_label = f"{BEST_MODEL} (đã tinh chỉnh)"
p_best = probas[best_label]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, t, ttl in [(axes[0], 0.5, "Ngưỡng 0,5"), (axes[1], THRESH, f"Ngưỡng cảnh báo {THRESH}")]:
    ConfusionMatrixDisplay(confusion_matrix(y_test, (p_best >= t).astype(int)),
                           display_labels=["Không PS", "Phá sản"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(ttl)
fig.suptitle(f"Hình 9. Ma trận nhầm lẫn — {BEST_MODEL} trên tập test")
savefig("fig09_confusion_matrix.png")

# %% [markdown]
# ## 7. Giải thích mô hình bằng SHAP

# %%
shap_top = None
model_step = best_pipe.named_steps["model"]
X_shap = X_test.copy()
if "scaler" in best_pipe.named_steps:
    X_shap = pd.DataFrame(best_pipe.named_steps["scaler"].transform(X_test), columns=FEATURES, index=X_test.index)

if HAS_SHAP:
    if BEST_MODEL == "LogisticRegression":
        explainer = shap.LinearExplainer(model_step, X_shap)
        sv = explainer(X_shap)
    else:
        explainer = shap.TreeExplainer(model_step)
        sv = explainer(X_shap)
        if sv.values.ndim == 3:  # một số phiên bản trả về cho cả 2 lớp
            sv = sv[:, :, 1]
    plt.figure()
    shap.summary_plot(sv.values, X_shap, max_display=10, show=False)
    plt.title("Hình 10. SHAP — 10 chỉ số ảnh hưởng mạnh nhất")
    savefig("fig10_shap_summary.png")

    shap_top = (pd.Series(np.abs(sv.values).mean(0), index=FEATURES)
                .sort_values(ascending=False).head(10))
    shap_top.to_csv(os.path.join(OUT, "table_shap_top10.csv"), header=["mean_abs_shap"])

    # Giải thích 1 doanh nghiệp phá sản thật bị mô hình cảnh báo với xác suất cao nhất
    idx_pos = np.where(y_test.values == 1)[0]
    i = int(idx_pos[np.argmax(p_best[idx_pos])])
    plt.figure()
    shap.plots.waterfall(sv[i], max_display=10, show=False)
    plt.title(f"Hình 11. Vì sao doanh nghiệp #{X_test.index[i]} bị cảnh báo (xác suất {p_best[i]:.2f})")
    savefig("fig11_shap_waterfall.png")
else:
    imp = getattr(model_step, "feature_importances_", None)
    if imp is None:
        imp = np.abs(model_step.coef_[0])
    shap_top = pd.Series(imp, index=FEATURES).sort_values(ascending=False).head(10)
    shap_top.to_csv(os.path.join(OUT, "table_feature_importance_top10.csv"), header=["importance"])
    print("Không có thư viện shap — dùng feature importance thay thế.")
shap_top

# %% [markdown]
# ## 8. Phân loại 3 mức rủi ro cho dashboard
# - **Thấp:** xác suất < ngưỡng cảnh báo
# - **Trung bình:** từ ngưỡng cảnh báo đến 0,5 (hoặc gấp đôi ngưỡng nếu ngưỡng ≥ 0,25)
# - **Cao:** trên mức đó

# %%
hi_cut = max(0.5, 2 * THRESH) if THRESH < 0.5 else min(0.95, THRESH + 0.2)
bins = [-0.001, THRESH, hi_cut, 1.0]
levels = pd.cut(p_best, bins=bins, labels=["Thấp", "Trung bình", "Cao"])
risk = (pd.DataFrame({"muc_rui_ro": levels, "pha_san_that": y_test.values})
        .groupby("muc_rui_ro", observed=False)
        .agg(so_doanh_nghiep=("pha_san_that", "size"), so_pha_san_that=("pha_san_that", "sum")))
risk["ty_le_pha_san_that"] = risk["so_pha_san_that"] / risk["so_doanh_nghiep"]
risk.to_csv(os.path.join(OUT, "table_risk_levels.csv"))
print("Ngưỡng:", [round(b, 2) for b in bins])
risk

# Danh sách cảnh báo mẫu (dùng cho thiết kế dashboard)
watch = (pd.DataFrame({"company_id": X_test.index, "xac_suat_pha_san": p_best.round(3),
                       "muc_rui_ro": levels, "pha_san_that": y_test.values})
         .sort_values("xac_suat_pha_san", ascending=False).head(20))
watch.to_csv(os.path.join(OUT, "table_top20_watchlist.csv"), index=False)

# %% [markdown]
# ## 9. Lưu kết quả và tải về
# Gửi file `results.zip` này cho trợ lý để viết Mục 4–6 của báo cáo.

# %%
bt = test_df[test_df.model.str.startswith(best_label) & (test_df.threshold == THRESH)].iloc[0]
summary = {
    "prep": prep_info,
    "best_model": BEST_MODEL, "best_strategy": BEST_STRAT, "best_params": search.best_params_,
    "cv_pr_auc_before_tuning": float(best_row["pr_auc_mean"]), "cv_pr_auc_after_tuning": float(search.best_score_),
    "threshold": THRESH, "risk_bins": [float(b) for b in bins],
    "test_best_at_threshold": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                                for k, v in bt.items()},
    "top10_features": list(shap_top.index), "libs": {"imblearn": HAS_IMBLEARN, "xgboost": HAS_XGB,
                                                     "lightgbm": HAS_LGBM, "shap": HAS_SHAP},
}
with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

shutil.make_archive("results", "zip", OUT)
print("Đã tạo results.zip")
if IN_COLAB:
    from google.colab import files
    files.download("results.zip")
