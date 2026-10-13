import os
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict
)

from sklearn.metrics import recall_score


# =====================================================
# 1. CAU HINH
# =====================================================

TARGET_SENSITIVITY = 0.98
RANDOM_STATE = 42

MODEL_DIR = "model_final"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "svm_rbf_final.joblib"
)


# =====================================================
# 2. NAP DU LIEU
# =====================================================

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

# 1 = malignant, 0 = benign
y = (data.target == 0).astype(int)

print("So mau:", len(X))
print("So dac trung:", X.shape[1])
print("So mau malignant:", int(y.sum()))
print("So mau benign:", int((y == 0).sum()))


# =====================================================
# 3. XAY DUNG PIPELINE
# =====================================================

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("select", SelectKBest(score_func=f_classif)),
    ("model", SVC(
        kernel="rbf",
        probability=True,
        random_state=RANDOM_STATE
    ))
])

param_grid = {
    "model__C": [0.1, 1, 10],
    "model__gamma": ["scale", 0.01, 0.1],
    "select__k": [10, 16, 20, "all"]
}


# =====================================================
# 4. CHON NGUONG BANG OOF
# =====================================================

threshold_cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=2026
)

inner_cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=2027
)

search_for_oof = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=inner_cv,
    n_jobs=-1,
    refit=True
)

print("\nDang tao du doan OOF de chon nguong...")
print("Qua trinh nay co the mat mot luc.")

oof_probabilities = cross_val_predict(
    search_for_oof,
    X,
    y,
    cv=threshold_cv,
    method="predict_proba",
    n_jobs=1
)[:, 1]


# =====================================================
# 5. TIM NGUONG CO SENSITIVITY DAT MUC TIEU
# =====================================================

thresholds = np.unique(oof_probabilities)

valid_thresholds = []

for threshold in thresholds:
    predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    sensitivity = recall_score(
        y,
        predictions,
        pos_label=1,
        zero_division=0
    )

    specificity = recall_score(
        y,
        predictions,
        pos_label=0,
        zero_division=0
    )

    if sensitivity >= TARGET_SENSITIVITY:
        valid_thresholds.append(
            (threshold, sensitivity, specificity)
        )

if valid_thresholds:
    # Uu tien specificity cao nhat trong cac nguong
    # dat sensitivity muc tieu tren OOF.
    threshold, oof_sensitivity, oof_specificity = max(
        valid_thresholds,
        key=lambda item: (item[2], item[0])
    )

    target_met = True

else:
    # Truong hop du phong neu khong co nguong dat muc tieu
    candidates = []

    for threshold in thresholds:
        predictions = (
            oof_probabilities >= threshold
        ).astype(int)

        sensitivity = recall_score(
            y,
            predictions,
            pos_label=1,
            zero_division=0
        )

        specificity = recall_score(
            y,
            predictions,
            pos_label=0,
            zero_division=0
        )

        candidates.append(
            (threshold, sensitivity, specificity)
        )

    threshold, oof_sensitivity, oof_specificity = max(
        candidates,
        key=lambda item: (item[1], item[2])
    )

    target_met = False


print("\nKet qua chon nguong OOF:")
print(f"Nguong: {threshold:.6f}")
print(f"OOF sensitivity: {oof_sensitivity:.4f}")
print(f"OOF specificity: {oof_specificity:.4f}")
print(f"Dat muc tieu OOF: {target_met}")


# =====================================================
# 6. HUAN LUYEN MO HINH CUOI CUNG TREN TOAN BO DU LIEU
# =====================================================

final_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

final_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=final_cv,
    n_jobs=-1,
    refit=True
)

print("\nDang huan luyen mo hinh cuoi cung...")

final_search.fit(X, y)

final_model = final_search.best_estimator_

selected_features = [
    X.columns[i]
    for i, selected in enumerate(
        final_model.named_steps["select"].get_support()
    )
    if selected
]


# =====================================================
# 7. LUU MO HINH VA THONG TIN
# =====================================================

artifact = {
    "model": final_model,
    "threshold": float(threshold),
    "target_sensitivity": TARGET_SENSITIVITY,
    "oof_sensitivity": float(oof_sensitivity),
    "oof_specificity": float(oof_specificity),
    "oof_target_met": bool(target_met),
    "feature_names": list(X.columns),
    "selected_features": selected_features,
    "positive_class": "malignant",
    "best_params": final_search.best_params_,
    "cv_best_auc": float(final_search.best_score_)
}

joblib.dump(artifact, MODEL_PATH)

print("\n" + "=" * 60)
print("DA LUU MO HINH THANH CONG")
print("=" * 60)
print("File:", os.path.abspath(MODEL_PATH))
print("Tham so tot nhat:", final_search.best_params_)
print("So dac trung duoc chon:", len(selected_features))
print("Nguong su dung:", round(threshold, 6))
print("AUC trung binh trong CV huan luyen:",
      round(final_search.best_score_, 4))
print("\nDanh sach dac trung:")
for feature in selected_features:
    print("-", feature)

print(
    "\nLuu y: OOF sensitivity/specificity la ket qua "
    "chon nguong tren du lieu huan luyen, "
    "khong phai ket qua kiem dinh doc lap."
)