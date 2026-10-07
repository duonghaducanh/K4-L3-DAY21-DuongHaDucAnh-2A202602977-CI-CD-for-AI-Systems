import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65

# Ty le lop duong (thu nhap > 50K) tham chieu cua bo du lieu goc.
# Dung cho Bonus 5 - canh bao lech lac du lieu.
REFERENCE_POSITIVE_RATE = 0.248


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    # 1. Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # 2. Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():

        # 3. Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # 4. Khoi tao va huan luyen GradientBoostingClassifier
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # 5. Du doan tren tap holdout va tinh chi so
        # f1_score tinh cho LOP DUONG (target = 1), khong dung average.
        preds = model.predict(X_eval)
        f1 = f1_score(y_eval, preds)
        acc = accuracy_score(y_eval, preds)

        # 6. Ghi nhan chi so vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        # 7. In ket qua ra man hinh
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")

        # ------------------------------------------------------------------
        # Bonus 5: Canh bao lech lac du lieu (data drift)
        # Tinh ty le lop duong trong tap huan luyen, canh bao neu lech qua
        # 5 diem phan tram so voi ty le tham chieu 24.8%.
        # ------------------------------------------------------------------
        positive_rate = float(y_train.mean())
        mlflow.log_metric("positive_rate", positive_rate)
        print(f"Ty le lop duong trong tap train: {positive_rate:.1%}")
        if abs(positive_rate - REFERENCE_POSITIVE_RATE) > 0.05:
            print(
                "CANH BAO LECH LAC DU LIEU: ty le lop duong "
                f"{positive_rate:.1%} lech qua 5 diem phan tram so voi "
                f"tham chieu {REFERENCE_POSITIVE_RATE:.1%}."
            )
        else:
            print(
                "Kiem tra phan phoi: ty le lop duong nam trong nguong cho phep "
                f"(tham chieu {REFERENCE_POSITIVE_RATE:.1%})."
            )

        # ------------------------------------------------------------------
        # Bonus 2: Dieu chinh nguong quyet dinh
        # model.predict() mac dinh dung nguong 0.5. Voi du lieu mat can bang,
        # day hiem khi la nguong toi uu. Quet nguong tu 0.1 den 0.9 buoc 0.05.
        # ------------------------------------------------------------------
        proba = model.predict_proba(X_eval)[:, 1]
        best_threshold, best_threshold_f1 = 0.5, float(f1)
        for thr in np.arange(0.1, 0.9 + 1e-9, 0.05):
            thr = float(round(thr, 2))
            f1_at_thr = f1_score(y_eval, (proba >= thr).astype(int))
            if f1_at_thr > best_threshold_f1:
                best_threshold, best_threshold_f1 = thr, float(f1_at_thr)

        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_threshold_f1", best_threshold_f1)
        print(
            f"Bonus 2 - Nguong mac dinh 0.50: F1={f1:.4f} | "
            f"Nguong toi uu {best_threshold:.2f}: F1={best_threshold_f1:.4f}"
        )

        # ------------------------------------------------------------------
        # Bonus 3: Bao cao precision / recall tu dong
        # ------------------------------------------------------------------
        cm = confusion_matrix(y_eval, preds)
        precision = precision_score(y_eval, preds)
        recall = recall_score(y_eval, preds)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        detail_lines = [
            "=== BAO CAO CHI TIET MO HINH (Bonus 3) ===",
            "",
            "Confusion matrix (hang = thuc te, cot = du doan):",
            "                 pred=0   pred=1",
            f"  actual=0     {cm[0][0]:>7d}  {cm[0][1]:>7d}",
            f"  actual=1     {cm[1][0]:>7d}  {cm[1][1]:>7d}",
            "",
            "Chi so theo lop:",
            f"  Lop 0 (thu nhap thap): precision={precision_score(y_eval, preds, pos_label=0):.4f} "
            f"recall={recall_score(y_eval, preds, pos_label=0):.4f}",
            f"  Lop 1 (thu nhap cao) : precision={precision:.4f} recall={recall:.4f}",
            "",
            f"  f1_score (lop duong) : {f1:.4f}",
            f"  accuracy             : {acc:.4f}",
            "",
            "Nhan xet ve chi phi sai lam:",
            "  Voi bai toan nay, bo sot mot nguoi thu nhap cao (recall thap) ton",
            "  kem hon gan nham mot nguoi thu nhap thap la thu nhap cao (precision",
            "  thap), vi muc tieu la tim ra khach hang tiem nang. Vi vay can uu tien",
            "  recall cua lop duong, va F1 (can bang precision/recall) la chi so",
            "  phu hop de lam nguong chat luong.",
        ]
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/detail.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(detail_lines))

        # 8. Luu metrics ra file outputs/report.json (doc boi GitHub Actions)
        report = {
            "f1_score": f1,
            "accuracy": acc,
            "positive_rate": positive_rate,
            "best_threshold": best_threshold,
            "best_threshold_f1": best_threshold_f1,
            "precision": precision,
            "recall": recall,
        }
        with open("outputs/report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # 9. Luu mo hinh ra file models/model.joblib (upload len cloud o Buoc 2)
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # 10. Tra ve f1
    return float(f1)


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
