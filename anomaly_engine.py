import json
from pathlib import Path
from collections import deque

import joblib
import numpy as np

from feature_extractors import (
    original_features,
    extract_power_features,
    extract_comm_features,
    thermal_features_v2
)


class AnomalyEngine:

    def __init__(self, models_dir="models"):

        self.models_dir = Path(models_dir)

        self.configs = {}
        self.models = {}
        self.scalers = {}

        for subsystem in [
            "attitude",
            "power",
            "communication",
            "thermal"
        ]:

            path = self.models_dir / subsystem

            with open(
                path / "config.json",
                "r"
            ) as f:
                self.configs[subsystem] = json.load(f)

            self.models[subsystem] = joblib.load(
                path / "ocsvm.joblib"
            )

            self.scalers[subsystem] = joblib.load(
                path / "scaler.joblib"
            )

        # Exact frozen window sizes
        self.buffers = {
            "attitude": deque(maxlen=20),
            "power": deque(maxlen=10),
            "communication": deque(maxlen=20),
            "thermal": deque(maxlen=10)
        }

        self.extractors = {
            "attitude": original_features,
            "power": extract_power_features,
            "communication": extract_comm_features,
            "thermal": thermal_features_v2
        }

        self._validate_artifacts()

    # ======================================================
    # VERIFY MODEL / FEATURE COMPATIBILITY
    # ======================================================

    def _validate_artifacts(self):

        expected = {
            "attitude": 30,
            "power": 21,
            "communication": 32,
            "thermal": 21
        }

        for subsystem, expected_count in expected.items():

            scaler_count = getattr(
                self.scalers[subsystem],
                "n_features_in_",
                None
            )

            if (
                scaler_count is not None
                and scaler_count != expected_count
            ):
                raise RuntimeError(
                    f"{subsystem}: scaler expects "
                    f"{scaler_count} features, "
                    f"but extractor should produce "
                    f"{expected_count}"
                )

    # ======================================================
    # ADD ONE STREAMING PACKET
    # ======================================================

    def update(self, packet):

        raw = packet["raw_telemetry"]

        # ---------------- ATTITUDE ----------------

        a = raw["attitude"]

        self.buffers["attitude"].append([
            a["X_Coarse_Spin"],
            a["Y_Coarse_Spin"],
            a["Z_Coarse_Spin"]
        ])

        # ---------------- POWER ----------------

        p = raw["power"]

        self.buffers["power"].append([
            p["Battery_Voltage"],
            p["Battery_Temp"]
        ])

        # ---------------- COMMUNICATION ----------------

        c = raw["communication"]

        self.buffers["communication"].append([
            c["Background_RSSI"],
            c["Last_RSSI"],
            c["NanoCom_Temp"]
        ])

        # ---------------- THERMAL ----------------

        t = raw["thermal"]

        self.buffers["thermal"].append([
            t["HD_Cam_Temp"],
            t["ACU2_Temp"],
            t["NanoMind_Temp"],
            t["SDR_Temp"],
            t["SEPP_Temp"]
        ])

        # Run whatever models have full windows
        results = {}

        for subsystem in self.buffers:

            results[subsystem] = (
                self._evaluate_subsystem(
                    subsystem
                )
            )

        return results

    # ======================================================
    # RUN ONE SUBSYSTEM MODEL
    # ======================================================

    def _evaluate_subsystem(
        self,
        subsystem
    ):

        buffer = self.buffers[subsystem]

        required = self.configs[
            subsystem
        ]["window_size"]

        if len(buffer) < required:

            return {
                "ready": False,
                "samples": len(buffer),
                "required": required,
                "score": None,
                "threshold": self.configs[
                    subsystem
                ]["threshold"],
                "is_anomaly": False
            }

        window = np.asarray(
            buffer,
            dtype=np.float32
        )

        # Extractors expect:
        # (number_of_windows, window_size, channels)
        batch = window[np.newaxis, :, :]

        extractor = self.extractors[
            subsystem
        ]

        features = extractor(batch)
    #     if subsystem == "power":
    #         print("\nPOWER RAW FEATURES:")
    #         print(features[0])

    #         print("POWER SCALED FEATURES:")
    #         print(
    #             self.scalers["power"]
    #             .transform(features)[0]
    # )

        scaler = self.scalers[
            subsystem
        ]

        model = self.models[
            subsystem
        ]

        expected = getattr(
            scaler,
            "n_features_in_",
            None
        )

        if (
            expected is not None
            and features.shape[1] != expected
        ):
            raise RuntimeError(
                f"{subsystem}: extractor produced "
                f"{features.shape[1]} features but "
                f"saved scaler expects {expected}"
            )

        scaled = scaler.transform(
            features
        )

        # Training convention:
        # larger score = more anomalous
        score = float(
            -model.decision_function(
                scaled
            )[0]
        )

        threshold = float(
            self.configs[
                subsystem
            ]["threshold"]
        )

        is_anomaly = (
            score >= threshold
        )

        return {
            "ready": True,
            "score": score,
            "threshold": threshold,
            "is_anomaly": bool(
                is_anomaly
            )
        }

    # ======================================================
    # OVERALL TRIGGER
    # ======================================================

    @staticmethod
    def any_anomaly(results):

        return any(
            result["ready"]
            and result["is_anomaly"]
            for result in results.values()
        )