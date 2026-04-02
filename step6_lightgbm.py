"""
Step 6. LightGBM 지도학습 분류기 — K-Means 클러스터 레이블 기반
================================================================
K-Means 클러스터 레이블(비지도 학습 결과)을 의사 타겟(pseudo-target)으로 사용해
LightGBM 지도학습 분류기를 학습합니다.

목적
  ① K-Means가 학습한 클러스터 구조가 지도학습으로도 재현 가능한지 검증
  ② 새 환자 입력에 대한 확률 기반 클래스 예측 제공
  ③ 피처 중요도 분석으로 클러스터 분리에 핵심 역할을 하는 변수 파악

출력
  - 분류 성능 리포트 (Accuracy, Weighted F1, 혼동행렬)
  - 피처 중요도 차트: outputs/figures/step6_feature_importance.png
  - 저장 모델: outputs/models/lgbm_rhinitis.pkl
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── 한글 폰트 ─────────────────────────────────────────
available_fonts = [f.name for f in fm.fontManager.ttflist]
if 'Malgun Gothic' in available_fonts:
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif 'AppleGothic' in available_fonts:
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

try:
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report,
        confusion_matrix, ConfusionMatrixDisplay
    )
    SKLEARN_OK = True
except ImportError as e:
    print(f"⚠️  패키지 미설치: {e}")
    print("   pip install lightgbm scikit-learn 후 재실행하세요.")
    SKLEARN_OK = False

print("=" * 65)
print("Step 6. LightGBM 지도학습 분류기 (K-Means 클러스터 레이블 기반)")
print("=" * 65)

# ── 데이터 로드 ──────────────────────────────────────
df = pd.read_csv('data/processed/rhinitis_clustered.csv')
print(f"\n데이터: {len(df):,}명 로드 완료")

FEATURE_COLS = [
    'has_asthma', 'has_atopic_derm', 'has_food_allergy',
    'food_allergy_count', 'rhinitis_onset_age', 'rhinitis_duration',
    'atopic_march', 'is_female',
]

X  = df[FEATURE_COLS].fillna(0).values
y  = df['cluster'].values          # 0 / 1 / 2
y_label = df['cluster_label'].values

label_names = {0: '호흡기 알레르기형', 1: '비염+천식 복합형', 2: '아토픽 마치형'}
LABEL_ORDER = ['호흡기 알레르기형', '비염+천식 복합형', '아토픽 마치형']

print(f"피처: {len(FEATURE_COLS)}개 / 클래스: {np.unique(y)}")
print("\n클래스 분포:")
for c in sorted(np.unique(y)):
    n = np.sum(y == c)
    print(f"  {label_names[c]:<18}: {n:>6,}명 ({n/len(y)*100:.1f}%)")

if not SKLEARN_OK:
    print("\n[종료] sklearn/lightgbm 없이는 학습 불가. 패키지를 설치하세요.")
    sys.exit(0)

# ── LightGBM 하이퍼파라미터 ──────────────────────────
LGB_PARAMS = {
    "objective":     "multiclass",
    "num_class":     3,
    "metric":        "multi_logloss",
    "n_estimators":  500,
    "learning_rate": 0.05,
    "num_leaves":    63,
    "max_depth":     -1,
    "min_child_samples": 50,
    "subsample":     0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":     0.1,
    "reg_lambda":    0.1,
    "random_state":  42,
    "n_jobs":        -1,
    "verbose":       -1,
}

# ── 5-Fold 교차검증 ───────────────────────────────────
print("\n" + "─" * 65)
print("5-Fold 층화 교차검증 실행 중...")
print("─" * 65)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc  = []
cv_f1   = []
fold_cm = np.zeros((3, 3), dtype=int)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    preds    = model.predict(X_val)
    acc      = accuracy_score(y_val, preds)
    f1       = f1_score(y_val, preds, average='weighted')
    cv_acc.append(acc)
    cv_f1.append(f1)
    fold_cm += confusion_matrix(y_val, preds, labels=[0, 1, 2])
    print(f"  Fold {fold}: Accuracy={acc:.4f}  Weighted-F1={f1:.4f}  "
          f"(best_iter={model.best_iteration_})")

print(f"\n  평균 Accuracy: {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")
print(f"  평균 F1-score: {np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f}")

# ── 전체 데이터로 최종 모델 학습 ─────────────────────
print("\n" + "─" * 65)
print("최종 모델 전체 데이터 학습...")
print("─" * 65)

final_model = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": 500})
final_model.fit(X, y, callbacks=[lgb.log_evaluation(-1)])

# 최종 모델로 전체 데이터 예측 (자기 검증)
y_pred_all = final_model.predict(X)
print("\n전체 데이터 분류 리포트:")
print(classification_report(
    y, y_pred_all,
    target_names=[label_names[c] for c in sorted(label_names)],
))

# ── 피처 중요도 ───────────────────────────────────────
importance = final_model.feature_importances_
feat_df = pd.DataFrame({
    "피처":  FEATURE_COLS,
    "중요도": importance,
}).sort_values("중요도", ascending=True)

# ── 시각화 ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (1) 피처 중요도 수평 막대
ax = axes[0]
colors = ["#4A90D9" if v >= feat_df["중요도"].quantile(0.6) else "#AACFE8"
          for v in feat_df["중요도"]]
ax.barh(feat_df["피처"], feat_df["중요도"], color=colors, edgecolor="white")
ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
ax.set_title("LightGBM 피처 중요도", fontsize=13)
ax.grid(axis="x", alpha=0.3)

# (2) 혼동행렬
ax = axes[1]
cmd = ConfusionMatrixDisplay(
    confusion_matrix=fold_cm,
    display_labels=[label_names[c] for c in [0, 1, 2]],
)
cmd.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"5-Fold 혼동행렬 합산\n(Acc={np.mean(cv_acc):.4f}, F1={np.mean(cv_f1):.4f})",
             fontsize=12)
plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)

plt.suptitle("LightGBM 분류기 — K-Means 클러스터 재현 성능", fontsize=14, y=1.02)
plt.tight_layout()

os.makedirs("outputs/figures", exist_ok=True)
save_path = "outputs/figures/step6_feature_importance.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\n차트 저장: {save_path}")

# ── 모델 저장 ────────────────────────────────────────
os.makedirs("outputs/models", exist_ok=True)
bundle = {
    "model":      final_model,
    "features":   FEATURE_COLS,
    "label_map":  label_names,
    "cv_acc":     float(np.mean(cv_acc)),
    "cv_f1":      float(np.mean(cv_f1)),
}
model_path = "outputs/models/lgbm_rhinitis.pkl"
joblib.dump(bundle, model_path)
print(f"모델 저장: {model_path}")

print("\n" + "=" * 65)
print("  Step 6 완료 — LightGBM 분류기 학습 결과 요약")
print("=" * 65)
print(f"  5-Fold Accuracy : {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")
print(f"  5-Fold F1       : {np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f}")
print(f"  Top-3 피처      : {', '.join(feat_df.tail(3)['피처'].tolist()[::-1])}")
print()
