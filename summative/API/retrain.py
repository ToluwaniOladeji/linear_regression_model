"""Retraining logic: validates an uploaded CSV, updates the model, and returns the new pipeline.

Used by prediction.py (the /retrain route). It can also be run by hand:
    python retrain.py new_rows.csv
"""
import copy
import sys

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import SGDRegressor

NUM_FEATURES = ["YearsCodePro", "YearsCode", "NumLanguages"]
CAT_FEATURES = ["EdLevel", "DevType", "Country", "RemoteWork", "OrgSize"]
FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "ConvertedCompYearly"
SALARY_FLOOR, SALARY_CEIL = 3000, 500000   # same filter as the notebook
SGD_EPOCHS = 20                            # only used if the saved model is an SGDRegressor


class RetrainError(ValueError):
    """The uploaded data cannot be used; the message is shown to the user."""


def clean_new_rows(new_df, countries):
    missing = [c for c in FEATURES + [TARGET] if c not in new_df.columns]
    if missing:
        raise RetrainError(f"Missing columns: {missing}")
    new = new_df[FEATURES + [TARGET]].copy()
    for c in NUM_FEATURES + [TARGET]:
        new[c] = pd.to_numeric(new[c], errors="coerce")
    new = new.dropna(subset=[TARGET])
    new = new[new[TARGET].between(SALARY_FLOOR, SALARY_CEIL)]
    new["Country"] = new["Country"].where(new["Country"].isin(countries), "Other")
    if new.empty:
        raise RetrainError("No usable rows after validation.")
    return new


def retrain(current, new_df, countries):
    """Return (updated_pipeline, summary). `current` is the loaded pipeline; it is not modified."""
    new = clean_new_rows(new_df, countries)
    old = getattr(current, "training_data_", None)
    if old is None:
        raise RetrainError("The saved model has no training data attached.")
    combined = pd.concat([old[FEATURES + [TARGET]], new], ignore_index=True)

    if isinstance(current.named_steps["model"], SGDRegressor):
        # SGD: continue from the saved weights with partial_fit on the NEW rows only.
        # The saved preprocessing (scaler, encoder) is reused as it is, not refitted.
        fresh = copy.deepcopy(current)
        X_new = fresh.named_steps["prep"].transform(new[FEATURES])
        y_new = new[TARGET] / 1000
        for _ in range(SGD_EPOCHS):
            fresh.named_steps["model"].partial_fit(X_new, y_new)
    else:
        # Any other model: refit from scratch on the original data plus the new rows.
        fresh = clone(current)                 # same preprocessing and model settings, unfitted
        fresh.fit(combined[FEATURES], combined[TARGET] / 1000)

    fresh.training_data_ = combined            # kept with the model for the next retrain
    err = fresh.predict(combined[FEATURES]) - combined[TARGET] / 1000
    summary = {"status": "retrained", "new_rows_used": int(len(new)), "total_rows": int(len(combined)),
               "train_rmse_thousand_usd": round(float((err ** 2).mean() ** 0.5), 2)}
    return fresh, summary


if __name__ == "__main__":
    # Manual use: python retrain.py new_rows.csv [model.pkl]
    csv_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "best_model.pkl"
    model = joblib.load(model_path)
    # countries: taken from the model's own training data
    known = model.training_data_["Country"].unique().tolist()
    updated, info = retrain(model, pd.read_csv(csv_path), known)
    joblib.dump(updated, model_path)
    print(info)