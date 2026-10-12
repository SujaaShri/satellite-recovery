import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from feature_extractors import extract_power_features


CSV_PATH = "uhf_telemetry.csv"
MODEL_DIR = Path("models/power")

POWER_COLS = [
    "Battery_Voltage",
    "Battery_Temp"
]

POWER_GAP_MINUTES = 3
POWER_WINDOW = 10


# ============================================================
# 1. LOAD REAL OPS-SAT TELEMETRY
# ============================================================

df = pd.read_csv(CSV_PATH)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"]
)

df = df.sort_values(
    "Timestamp"
).reset_index(drop=True)


# ============================================================
# 2. KEEP ONLY PAIRED POWER OBSERVATIONS
# ============================================================

power_df = (
    df[
        [
            "Timestamp",
            "Battery_Voltage",
            "Battery_Temp"
        ]
    ]
    .dropna()
    .copy()
)

print(
    "Complete power observations:",
    len(power_df)
)


# ============================================================
# 3. RECREATE POWER SESSIONS
# ============================================================

power_df["gap_minutes"] = (
    power_df["Timestamp"]
    .diff()
    .dt.total_seconds()
    / 60.0
)

power_df["new_session"] = (
    power_df["gap_minutes"].isna()
    |
    (
        power_df["gap_minutes"]
        > POWER_GAP_MINUTES
    )
)

power_df["session_id"] = (
    power_df["new_session"]
    .cumsum()
)

all_sessions = [
    group.reset_index(drop=True)
    for _, group
    in power_df.groupby("session_id")
]


# ============================================================
# 4. KEEP SESSIONS THAT CAN FORM 10-SAMPLE WINDOWS
# ============================================================

usable_sessions = [
    session
    for session in all_sessions
    if len(session) >= POWER_WINDOW
]

usable_sessions = sorted(
    usable_sessions,
    key=lambda x: x["Timestamp"].iloc[0]
)

print(
    "All sessions:",
    len(all_sessions)
)

print(
    "Usable sessions:",
    len(usable_sessions)
)


# ============================================================
# 5. EXACT CHRONOLOGICAL SPLIT FROM FINAL POWER NOTEBOOK
#
# Expected:
# Train = 18
# Val   = 4
# Test  = 6
# ============================================================

power_train_sessions = usable_sessions[:18]

power_val_sessions = usable_sessions[18:22]

power_test_sessions = usable_sessions[22:]


print(
    "Train / Val / Test sessions:",
    len(power_train_sessions),
    len(power_val_sessions),
    len(power_test_sessions)
)


# ============================================================
# 6. CREATE RAW WINDOWS
# ============================================================

def make_power_windows(sessions):

    windows = []

    for session in sessions:

        values = (
            session[POWER_COLS]
            .values
            .astype(np.float32)
        )

        for start in range(
            len(values)
            - POWER_WINDOW
            + 1
        ):

            windows.append(
                values[
                    start:
                    start + POWER_WINDOW
                ]
            )

    return np.asarray(
        windows,
        dtype=np.float32
    )


X_train = make_power_windows(
    power_train_sessions
)

X_val = make_power_windows(
    power_val_sessions
)

X_test = make_power_windows(
    power_test_sessions
)


print(
    "Window shapes:"
)

print(
    "Train:",
    X_train.shape
)

print(
    "Val:  ",
    X_val.shape
)

print(
    "Test: ",
    X_test.shape
)


# ============================================================
# 7. SANITY CHECK
#
# These MUST match the final notebook.
# ============================================================

expected_train = (124, 10, 2)
expected_val = (22, 10, 2)
expected_test = (13, 10, 2)


if X_train.shape != expected_train:
    raise RuntimeError(
        f"Train mismatch: "
        f"{X_train.shape} != {expected_train}"
    )

if X_val.shape != expected_val:
    raise RuntimeError(
        f"Validation mismatch: "
        f"{X_val.shape} != {expected_val}"
    )

if X_test.shape != expected_test:
    raise RuntimeError(
        f"Test mismatch: "
        f"{X_test.shape} != {expected_test}"
    )


print(
    "Power reconstruction matches notebook."
)


# ============================================================
# 8. LOAD SAVED ARTIFACTS
# ============================================================

model = joblib.load(
    MODEL_DIR / "ocsvm.joblib"
)

scaler = joblib.load(
    MODEL_DIR / "scaler.joblib"
)

with open(
    MODEL_DIR / "config.json",
    "r"
) as f:

    config = json.load(f)


threshold = float(
    config["threshold"]
)


print(
    "Saved threshold:",
    threshold
)

print(
    "Scaler feature count:",
    scaler.n_features_in_
)


# ============================================================
# 9. TEST FUNCTION
# ============================================================

def evaluate(name, X):

    features = extract_power_features(
        X
    )

    if features.shape[1] != 21:
        raise RuntimeError(
            "Power extractor did not produce "
            f"21 features: {features.shape}"
        )

    scaled = scaler.transform(
        features
    )

    decision = (
        model
        .decision_function(scaled)
        .ravel()
    )

    anomaly_score = -decision

    prediction = (
        anomaly_score >= threshold
    )

    print()
    print(name)
    print(
        "Windows:",
        len(X)
    )

    print(
        "Anomalies:",
        int(prediction.sum())
    )

    print(
        "Normal:",
        int((~prediction).sum())
    )

    print(
        "Anomaly rate:",
        round(
            prediction.mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Score min:",
        round(
            float(
                anomaly_score.min()
            ),
            6
        )
    )

    print(
        "Score median:",
        round(
            float(
                np.median(
                    anomaly_score
                )
            ),
            6
        )
    )

    print(
        "Score max:",
        round(
            float(
                anomaly_score.max()
            ),
            6
        )
    )

    return anomaly_score, prediction


# ============================================================
# 10. RUN ON REAL OPS-SAT
# ============================================================

train_scores, train_pred = evaluate(
    "REAL TRAIN NOMINAL",
    X_train
)

val_scores, val_pred = evaluate(
    "REAL VALIDATION NOMINAL",
    X_val
)

test_scores, test_pred = evaluate(
    "REAL CHRONOLOGICAL TEST",
    X_test
)