import numpy as np


# ============================================================
# TRAINING CONSTANTS RECOVERED FROM FINAL NOTEBOOK
# ============================================================

# Power
V_IQR = 310.0
T_IQR = 81.5

# Communication
LAST_MEDIAN = -100.0
BG_IQR = 12.0
TEMP_IQR = 143.0

# Thermal
THERMAL_COLS = [
    "HD_Cam_Temp",
    "ACU2_Temp",
    "NanoMind_Temp",
    "SDR_Temp",
    "SEPP_Temp"
]

THERMAL_MODEL_COLS = [
    "ACU2_Temp",
    "NanoMind_Temp",
    "SEPP_Temp"
]

THERMAL_IQRS = {
    "HD_Cam_Temp": 15.5,
    "ACU2_Temp": 115.75,
    "NanoMind_Temp": 63.25,
    "SDR_Temp": 32.0,
    "SEPP_Temp": 29.75
}


# ============================================================
# ATTITUDE
# Exact final original_features()
# 30 features
# ============================================================

def original_features(X):

    X = np.asarray(X, dtype=np.float32)

    X_deg = np.degrees(X)

    spin = np.sqrt(
        np.sum(
            X_deg ** 2,
            axis=2
        )
    )

    delta_xyz = np.diff(
        X_deg,
        axis=1
    )

    delta_spin = np.diff(
        spin,
        axis=1
    )

    angular_change = np.sqrt(
        np.sum(
            delta_xyz ** 2,
            axis=2
        )
    )

    t = np.arange(
        X.shape[1],
        dtype=np.float32
    )

    t_centered = t - t.mean()

    denominator = np.sum(
        t_centered ** 2
    )

    rows = []

    for i in range(len(X)):

        row = []

        window = X_deg[i]
        spin_w = spin[i]
        delta_spin_w = delta_spin[i]
        angular_w = angular_change[i]

        for axis in range(3):

            v = window[:, axis]

            slope = (
                np.sum(
                    t_centered
                    *
                    (
                        v - v.mean()
                    )
                )
                /
                denominator
            )

            row.extend([
                v.mean(),
                v.std(),
                np.max(np.abs(v)),
                slope,
                v[-1] - v[0],
                np.ptp(v)
            ])

        spin_slope = (
            np.sum(
                t_centered
                *
                (
                    spin_w - spin_w.mean()
                )
            )
            /
            denominator
        )

        row.extend([
            spin_w.mean(),
            spin_w.std(),
            spin_w.max(),
            spin_w.min(),
            np.ptp(spin_w),
            spin_slope,
            spin_w[-1] - spin_w[0],
            np.mean(
                np.abs(delta_spin_w)
            ),
            np.max(
                np.abs(delta_spin_w)
            ),
            angular_w.mean(),
            angular_w.std(),
            angular_w.max()
        ])

        rows.append(row)

    return np.asarray(
        rows,
        dtype=np.float32
    )


# ============================================================
# POWER
# Exact final extract_power_features()
# 21 features
# ============================================================

def linear_slope(x):

    t = np.arange(len(x))

    return np.polyfit(
        t,
        x,
        1
    )[0]


def extract_power_features(X):

    X = np.asarray(X, dtype=np.float32)

    features = []

    for window in X:

        v = window[:, 0].astype(float)
        t = window[:, 1].astype(float)

        dv = np.diff(v)
        dt = np.diff(t)

        v_range = np.ptp(v)
        t_range = np.ptp(t)

        v_slope = linear_slope(v)
        t_slope = linear_slope(t)

        v_net = v[-1] - v[0]
        t_net = t[-1] - t[0]

        if (
            np.std(v) > 1e-8
            and
            np.std(t) > 1e-8
        ):
            corr = np.corrcoef(v, t)[0, 1]
        else:
            corr = 0.0

        if not np.isfinite(corr):
            corr = 0.0

        normalized_v_decline = (
            max(0.0, -v_net)
            /
            max(V_IQR, 1e-8)
        )

        normalized_t_rise = (
            max(0.0, t_net)
            /
            max(T_IQR, 1e-8)
        )

        features.append([
            np.mean(v),
            np.std(v),
            np.min(v),
            np.max(v),
            v_range,
            v_slope,
            v_net,
            np.mean(np.abs(dv)),
            np.max(np.abs(dv)),

            np.mean(t),
            np.std(t),
            np.min(t),
            np.max(t),
            t_range,
            t_slope,
            t_net,
            np.mean(np.abs(dt)),
            np.max(np.abs(dt)),

            corr,
            normalized_v_decline,
            normalized_t_rise
        ])

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# COMMUNICATION
# Exact final feature logic
# 32 features
# ============================================================

def safe_slope(x):

    x = np.asarray(x, dtype=float)

    if len(x) < 2:
        return 0.0

    t = np.arange(len(x), dtype=float)

    if np.std(x) == 0:
        return 0.0

    return np.polyfit(
        t,
        x,
        1
    )[0]


def safe_stats(x):

    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        return [
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ]

    d = np.diff(x)

    return [
        np.mean(x),
        np.std(x),
        np.min(x),
        np.max(x),
        np.ptp(x),
        safe_slope(x),
        x[-1] - x[0],
        np.mean(np.abs(d)) if len(d) else 0.0,
        np.max(np.abs(d)) if len(d) else 0.0
    ]


def safe_corr(a, b):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if (
        len(a) < 2
        or np.std(a) == 0
        or np.std(b) == 0
    ):
        return 0.0

    corr = np.corrcoef(a, b)[0, 1]

    if not np.isfinite(corr):
        return 0.0

    return corr


def extract_comm_features(windows):

    windows = np.asarray(
        windows,
        dtype=np.float32
    )

    features = []

    for window in windows:

        bg = window[:, 0].astype(float)
        last = window[:, 1].astype(float)
        temp = window[:, 2].astype(float)

        row = []

        # Background RSSI: 9
        row.extend(
            safe_stats(bg)
        )

        # Valid Last_RSSI: 7
        valid_mask = last != 0
        valid_last = last[valid_mask]

        valid_fraction = np.mean(valid_mask)

        if len(valid_last) > 0:

            last_mean = np.mean(valid_last)
            last_std = np.std(valid_last)
            last_min = np.min(valid_last)
            last_max = np.max(valid_last)

            median_deviation = np.mean(
                np.abs(
                    valid_last - LAST_MEDIAN
                )
            )

            if len(valid_last) > 1:
                last_mean_abs_delta = np.mean(
                    np.abs(
                        np.diff(valid_last)
                    )
                )
            else:
                last_mean_abs_delta = 0.0

        else:

            last_mean = LAST_MEDIAN
            last_std = 0.0
            last_min = LAST_MEDIAN
            last_max = LAST_MEDIAN
            median_deviation = 0.0
            last_mean_abs_delta = 0.0

        row.extend([
            valid_fraction,
            last_mean,
            last_std,
            last_min,
            last_max,
            median_deviation,
            last_mean_abs_delta
        ])

        # Signal/background margin: 4
        if np.any(valid_mask):

            margin = (
                last[valid_mask]
                -
                bg[valid_mask]
            )

            margin_mean = np.mean(margin)
            margin_min = np.min(margin)
            margin_std = np.std(margin)
            margin_slope = safe_slope(margin)

        else:

            margin_mean = 0.0
            margin_min = 0.0
            margin_std = 0.0
            margin_slope = 0.0

        row.extend([
            margin_mean,
            margin_min,
            margin_std,
            margin_slope
        ])

        # NanoCom temperature: 9
        row.extend(
            safe_stats(temp)
        )

        # Cross-channel: 3
        bg_temp_corr = safe_corr(
            bg,
            temp
        )

        normalized_bg_rise = (
            (bg[-1] - bg[0])
            /
            BG_IQR
        )

        normalized_temp_rise = (
            (temp[-1] - temp[0])
            /
            TEMP_IQR
        )

        row.extend([
            bg_temp_corr,
            normalized_bg_rise,
            normalized_temp_rise
        ])

        features.append(row)

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# THERMAL V2
# Exact final 21-feature representation
# ============================================================

def thermal_features_v2(X):

    X = np.asarray(
        X,
        dtype=np.float32
    )

    features = []

    for window in X:

        row = []
        channel_data = {}

        for col in THERMAL_MODEL_COLS:

            j = THERMAL_COLS.index(col)

            x = (
                window[:, j]
                .astype(float)
            )

            channel_data[col] = x

            dx = np.diff(x)

            row.extend([
                np.std(x),
                np.max(x) - np.min(x),
                safe_slope(x),
                x[-1] - x[0],
                np.mean(np.abs(dx)),
                np.max(np.abs(dx))
            ])

        acu = channel_data[
            "ACU2_Temp"
        ]

        nano = channel_data[
            "NanoMind_Temp"
        ]

        sepp = channel_data[
            "SEPP_Temp"
        ]

        acu_nano_corr = safe_corr(
            acu,
            nano
        )

        acu_centered = (
            acu - acu[0]
        ) / max(
            THERMAL_IQRS["ACU2_Temp"],
            1e-6
        )

        nano_centered = (
            nano - nano[0]
        ) / max(
            THERMAL_IQRS["NanoMind_Temp"],
            1e-6
        )

        divergence = np.mean(
            np.abs(
                acu_centered
                -
                nano_centered
            )
        )

        dynamic_norm = np.column_stack([
            (
                acu - acu[0]
            )
            /
            max(
                THERMAL_IQRS["ACU2_Temp"],
                1e-6
            ),

            (
                nano - nano[0]
            )
            /
            max(
                THERMAL_IQRS["NanoMind_Temp"],
                1e-6
            ),

            (
                sepp - sepp[0]
            )
            /
            max(
                THERMAL_IQRS["SEPP_Temp"],
                1e-6
            )
        ])

        dynamic_spread = np.mean(
            np.max(
                dynamic_norm,
                axis=1
            )
            -
            np.min(
                dynamic_norm,
                axis=1
            )
        )

        row.extend([
            acu_nano_corr,
            divergence,
            dynamic_spread
        ])

        features.append(row)

    return np.asarray(
        features,
        dtype=np.float32
    )