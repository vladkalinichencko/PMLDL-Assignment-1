import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "titanic.csv"


def ensure_dataset() -> Path:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH


def load_dataset() -> pd.DataFrame:
    csv_path = ensure_dataset()
    df = pd.read_csv(csv_path)

    columns = ["Pclass", "Sex", "Age", "Fare", "Survived"]
    df = df[columns]

    df = df.dropna(subset=["Survived", "Pclass", "Sex"])
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    df["Pclass"] = df["Pclass"].astype(int)
    df["Sex"] = df["Sex"].astype(str).str.strip().str.lower()

    return df


def train_model(df: pd.DataFrame, seed: int = 42) -> tuple[CatBoostClassifier, float]:
    X = df[["Pclass", "Sex", "Age", "Fare"]]
    y = df["Survived"]

    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = int(0.8 * len(df))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = CatBoostClassifier(
        iterations=150,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(X_train, y_train, cat_features=[1])

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    acc = float((preds == y_test.to_numpy()).mean())

    return model, acc


def save_artifacts(model: CatBoostClassifier, acc: float) -> None:
    models_dir = Path(__file__).resolve().parents[2] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "titanic_model.cbm"
    model.save_model(model_path)

    meta = {
        "feature_names": ["Pclass", "Sex", "Age", "Fare"],
        "target_names": ["Died", "Survived"],
        "model": "CatBoostClassifier",
        "dataset": "titanic_real",
        "metrics": {"accuracy": acc},
    }

    with open(models_dir / "metadata.json", "w") as f:
        json.dump(meta, f)

    print(f"Saved model to {model_path}, acc={acc:.3f}")


def main() -> None:
    df = load_dataset()
    model, acc = train_model(df, seed=42)
    save_artifacts(model, acc)


if __name__ == "__main__":
    main()
