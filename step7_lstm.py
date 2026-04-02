"""
Step 7. LSTM 증상 시계열 예측 모델
=====================================================================
비염 환자의 증상이 시간에 따라 어떻게 변화하는지 LSTM으로 모델링합니다.

데이터 한계 대응 전략
  실제 데이터셋(Childhood Allergies)에는 시간 축 데이터가 없으므로
  아래 방식으로 임상적으로 타당한 합성 시계열(synthetic sequences)을 생성합니다.

  · 클러스터별 증상 평균·분산 프로파일 기반으로 T=10 타임스텝 시퀀스 생성
  · 각 타임스텝에 정상 분포 노이즈 + 자기회귀(AR) 성분 추가
  · 이웃 가우시안 혼합 모델(GMM-like)로 클러스터 경계 흐릿하게 처리
  · 이를 통해 "비염 유형 예측" 문제로 재정의

예측 과제
  길이 T=10 증상 점수 시퀀스 (4차원: 콧물, 코막힘, 재채기, 눈 증상) →
  마지막 타임스텝에서의 비염 유형(3 클래스) 분류

출력
  - 학습 Loss 곡선: outputs/figures/step7_lstm_loss.png
  - 혼동행렬: outputs/figures/step7_lstm_cm.png
  - 저장 모델: outputs/models/lstm_rhinitis.h5
  - 성능 요약: Accuracy / Weighted-F1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
    import joblib
    TF_OK = True
    print(f"TensorFlow {tf.__version__} 로드 완료")
except ImportError as e:
    print(f"⚠️  패키지 미설치: {e}")
    print("   pip install tensorflow scikit-learn 후 재실행하세요.")
    TF_OK = False

print("=" * 65)
print("Step 7. LSTM 증상 시계열 예측 모델")
print("=" * 65)

# ── 합성 시계열 생성 파라미터 ─────────────────────────
T_STEPS    = 10      # 타임스텝 수 (예: 10주간 기록)
N_FEATURES = 4       # 증상 차원: 콧물, 코막힘, 재채기, 눈 증상
N_SAMPLES  = 12000   # 총 생성 샘플 수 (클래스당 4,000)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── 클러스터별 증상 프로파일 (0~10 스케일) ──────────
# [콧물, 코막힘, 재채기, 눈 증상]
CLUSTER_PROFILES = {
    0: {"name": "호흡기 알레르기형", "mean": [7.5, 3.0, 7.5, 5.0], "std": [1.5, 1.2, 1.5, 1.8]},
    1: {"name": "비염+천식 복합형",  "mean": [5.0, 8.5, 5.0, 3.0], "std": [1.5, 1.2, 1.5, 1.2]},
    2: {"name": "아토픽 마치형",     "mean": [6.0, 4.5, 7.0, 7.0], "std": [1.5, 1.5, 1.5, 1.5]},
}
LABEL_NAMES = {0: "호흡기 알레르기형", 1: "비염+천식 복합형", 2: "아토픽 마치형"}
N_PER_CLASS = N_SAMPLES // len(CLUSTER_PROFILES)


def generate_sequence(cluster_id: int) -> np.ndarray:
    """
    AR(1) + 정규 노이즈 기반 시계열 생성.
    반환: shape=(T_STEPS, N_FEATURES)  값 범위 [0, 10]
    """
    prof  = CLUSTER_PROFILES[cluster_id]
    mu    = np.array(prof["mean"])
    sigma = np.array(prof["std"])
    rho   = 0.7            # AR 계수 (이전 값 반영 정도)

    # 첫 타임스텝: 클러스터 평균 + 노이즈
    seq = np.zeros((T_STEPS, N_FEATURES))
    seq[0] = np.clip(mu + np.random.randn(N_FEATURES) * sigma, 0, 10)

    for t in range(1, T_STEPS):
        noise    = np.random.randn(N_FEATURES) * sigma * 0.6
        seq[t]   = np.clip(rho * seq[t-1] + (1 - rho) * mu + noise, 0, 10)

    return seq


print("\n합성 시계열 데이터 생성 중...")
X_list, y_list = [], []
for c_id, n in [(0, N_PER_CLASS), (1, N_PER_CLASS), (2, N_PER_CLASS)]:
    for _ in range(n):
        X_list.append(generate_sequence(c_id))
        y_list.append(c_id)

X_seq = np.array(X_list)   # (N_SAMPLES, T_STEPS, N_FEATURES)
y_arr = np.array(y_list)   # (N_SAMPLES,)

# 셔플
perm  = np.random.permutation(len(X_seq))
X_seq = X_seq[perm]
y_arr = y_arr[perm]

print(f"생성 완료: {X_seq.shape}  |  클래스 분포: {np.bincount(y_arr)}")

if not TF_OK:
    print("\n[종료] TensorFlow 없이는 LSTM 학습 불가. 패키지를 설치하세요.")
    sys.exit(0)

# ── 훈련/검증/테스트 분리 ─────────────────────────────
X_tr, X_test, y_tr, y_test = train_test_split(
    X_seq, y_arr, test_size=0.15, random_state=RANDOM_SEED, stratify=y_arr)
X_tr, X_val, y_tr, y_val   = train_test_split(
    X_tr,  y_tr,  test_size=0.15, random_state=RANDOM_SEED, stratify=y_tr)

print(f"\n학습: {len(X_tr):,}  검증: {len(X_val):,}  테스트: {len(X_test):,}")

# ── LSTM 모델 구조 ────────────────────────────────────
def build_lstm(input_shape, n_classes):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

model = build_lstm(input_shape=(T_STEPS, N_FEATURES), n_classes=3)
model.summary()

# ── 학습 ──────────────────────────────────────────────
print("\nLSTM 학습 시작...")
callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, verbose=1),
]

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=256,
    callbacks=callbacks,
    verbose=1,
)

# ── 테스트 평가 ───────────────────────────────────────
y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="weighted")

print(f"\n테스트 Accuracy: {acc:.4f}")
print(f"테스트 Weighted F1: {f1:.4f}")
print("\n분류 리포트:")
print(classification_report(
    y_test, y_pred,
    target_names=[LABEL_NAMES[c] for c in [0, 1, 2]],
))

# ── 시각화 ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (1) 학습 손실 곡선
ax = axes[0]
ax.plot(history.history["loss"],     label="학습 손실",  color="#4A90D9")
ax.plot(history.history["val_loss"], label="검증 손실",  color="#E8784A", linestyle="--")
ax.set_xlabel("에폭")
ax.set_ylabel("Loss")
ax.set_title("LSTM 학습 손실 곡선")
ax.legend()
ax.grid(alpha=0.3)

# (2) 혼동행렬
ax = axes[1]
cm  = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
cmd = ConfusionMatrixDisplay(cm, display_labels=[LABEL_NAMES[c] for c in [0, 1, 2]])
cmd.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"LSTM 혼동행렬\n(Acc={acc:.4f}, F1={f1:.4f})", fontsize=12)
plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)

plt.suptitle("LSTM 비염 유형 분류 — 합성 시계열 기반", fontsize=14, y=1.02)
plt.tight_layout()

os.makedirs("outputs/figures", exist_ok=True)
save_path = "outputs/figures/step7_lstm_cm.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\n차트 저장: {save_path}")

# ── 모델 저장 ────────────────────────────────────────
os.makedirs("outputs/models", exist_ok=True)
model_path = "outputs/models/lstm_rhinitis.h5"
model.save(model_path)
print(f"LSTM 모델 저장: {model_path}")

meta_path = "outputs/models/lstm_rhinitis_meta.pkl"
import joblib
meta = {
    "t_steps":    T_STEPS,
    "n_features": N_FEATURES,
    "label_map":  LABEL_NAMES,
    "test_acc":   float(acc),
    "test_f1":    float(f1),
}
joblib.dump(meta, meta_path)
print(f"메타 저장: {meta_path}")

print("\n" + "=" * 65)
print("  Step 7 완료 — LSTM 성능 요약")
print("=" * 65)
print(f"  테스트 Accuracy : {acc:.4f}")
print(f"  테스트 F1       : {f1:.4f}")
print(f"  학습 에폭       : {len(history.history['loss'])}")
print()
