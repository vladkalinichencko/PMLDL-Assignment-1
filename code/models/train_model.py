import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


def generate_data(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    pclass = rng.integers(1, 4, size=n)
    sex = rng.choice(["male", "female"], size=n)

    age = np.clip(rng.normal(30, 14, size=n), 0, 80)

    fare_base = 15 * (4 - pclass) ** 1.3
    fare_noise = rng.normal(0, 8, size=n)
    fare = np.clip(fare_base + fare_noise, 0, None)

    logit = (
        -1.0
        + 1.1 * (sex == "female")
        + 0.6 * (pclass == 1)
        + 0.2 * (pclass == 2)
        - 0.02 * (age - 30)
        + 0.01 * (fare - 15)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    survived = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "Pclass": pclass,
            "Sex": sex,
            "Age": age.round(1),
            "Fare": fare.round(2),
            "Survived": survived,
        }
    )


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
        "dataset": "synthetic_titanic",
        "metrics": {"accuracy": acc},
    }

    with open(models_dir / "metadata.json", "w") as f:
        json.dump(meta, f)

    print(f"Saved model to {model_path}, acc={acc:.3f}")


def main() -> None:
    df = generate_data(n=2000, seed=42)
    model, acc = train_model(df, seed=42)
    save_artifacts(model, acc)


if __name__ == "__main__":
    main()
