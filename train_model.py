from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATH = Path("data/iris.csv")
MODEL_PATH = Path("model.pkl")
META_PATH = Path("model_meta.json")

def main():
    df = pd.read_csv(DATA_PATH)

    # Features + target (use clean column names)
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[features]
    y_text = df["species"]

    # Encode labels to 0/1/2 and keep class names
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    class_names = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:\n",
          classification_report(y_test, y_pred, target_names=class_names))

    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model → {MODEL_PATH.resolve()}")

    meta = {
        "model_type": "LogisticRegression (with StandardScaler)",
        "problem_type": "classification",
        "features": features,
        "class_names": class_names,
        "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "metrics": {"accuracy": float(acc)},
        "sklearn_version": sklearn_version,
        "has_predict_proba": True,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved metadata → {META_PATH.resolve()}")

if __name__ == "__main__":
    main()
