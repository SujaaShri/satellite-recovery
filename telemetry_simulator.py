import time
from datetime import datetime

import numpy as np
import pandas as pd


class SatelliteTelemetrySimulator:

    def __init__(
        self,
        csv_path="uhf_telemetry.csv",
        interval=0.5,
        seed=42
    ):
        self.interval = interval
        self.rng = np.random.default_rng(seed)
        self.step = 0

        self.active_faults = {}

        self.df = pd.read_csv(csv_path)

        self.df["Timestamp"] = pd.to_datetime(
            self.df["Timestamp"]
        )

        self.df = (
            self.df
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )

        # =====================================================
        # RAW CHANNEL DEFINITIONS
        # =====================================================

        self.ATTITUDE_COLS = [
            "X_Coarse_Spin",
            "Y_Coarse_Spin",
            "Z_Coarse_Spin"
        ]

        self.POWER_COLS = [
            "Battery_Voltage",
            "Battery_Temp"
        ]

        self.COMM_COLS = [
            "Background_RSSI",
            "Last_RSSI",
            "NanoCom_Temp"
        ]

        self.THERMAL_COLS = [
            "HD_Cam_Temp",
            "ACU2_Temp",
            "NanoMind_Temp",
            "SDR_Temp",
            "SEPP_Temp"
        ]

        # =====================================================
        # BUILD REAL TRAINING-REGIME TELEMETRY SOURCES
        # =====================================================

        self.attitude_sessions = self._build_training_sessions(
            columns=self.ATTITUDE_COLS,
            gap_minutes=3,
            window_size=20,
            train_count=None,
            train_ratio=0.65
        )

        # Exact final Power configuration:
        # 28 usable sessions -> first 18 training sessions
        self.power_sessions = self._build_training_sessions(
            columns=self.POWER_COLS,
            gap_minutes=3,
            window_size=10,
            train_count=18
        )

        # Final Communication:
        # 142 usable -> 92 training sessions
        self.communication_sessions = self._build_training_sessions(
            columns=self.COMM_COLS,
            gap_minutes=2,
            window_size=20,
            train_count=92
        )

        # Final Thermal:
        # 25 usable -> 16 training sessions
        self.thermal_sessions = self._build_training_sessions(
            columns=self.THERMAL_COLS,
            gap_minutes=3,
            window_size=10,
            train_count=16
        )

        if not self.attitude_sessions:
            raise RuntimeError(
                "No usable attitude training sessions found."
            )

        if not self.power_sessions:
            raise RuntimeError(
                "No usable power training sessions found."
            )

        if not self.communication_sessions:
            raise RuntimeError(
                "No usable communication training sessions found."
            )

        if not self.thermal_sessions:
            raise RuntimeError(
                "No usable thermal training sessions found."
            )

        # =====================================================
        # SESSION REPLAY STATE
        # =====================================================

        self.att_session_index = 0
        self.att_row_index = 0

        self.power_session_index = 0
        self.power_row_index = 0

        self.comm_session_index = 0
        self.comm_row_index = 0

        self.thermal_session_index = 0
        self.thermal_row_index = 0

        # =====================================================
        # AGENT-FACING SIMULATED STATE
        # =====================================================

        self.battery_soc = 82.0

        self.payload_subsystem = (
            "ACTIVE_HIGH_DRAW"
        )

        self.frc_horizon_minutes = 45

        # =====================================================
        # CURRENT RAW TELEMETRY
        # =====================================================

        self._load_next_nominal_values()

    # =========================================================
    # SESSION CONSTRUCTION
    # =========================================================

    def _build_training_sessions(
        self,
        columns,
        gap_minutes,
        window_size,
        train_count=None,
        train_ratio=0.65
    ):
        data = (
            self.df[
                ["Timestamp"] + columns
            ]
            .dropna(subset=columns)
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )

        data["gap_minutes"] = (
            data["Timestamp"]
            .diff()
            .dt.total_seconds()
            / 60.0
        )

        data["new_session"] = (
            data["gap_minutes"].isna()
            |
            (
                data["gap_minutes"]
                > gap_minutes
            )
        )

        data["session_id"] = (
            data["new_session"]
            .cumsum()
        )

        sessions = []

        for _, group in data.groupby(
            "session_id"
        ):
            group = group.reset_index(
                drop=True
            )

            if len(group) >= window_size:
                sessions.append(group)

        sessions = sorted(
            sessions,
            key=lambda x:
                x["Timestamp"].iloc[0]
        )

        if train_count is not None:
            return sessions[:train_count]

        n_train = int(
            len(sessions) * train_ratio
        )

        return sessions[:n_train]

    # =========================================================
    # GET NEXT ROW FROM A SUBSYSTEM SESSION
    #
    # IMPORTANT:
    # We do NOT concatenate the end of one session directly
    # into the beginning of another detector window.
    #
    # Instead, once a session finishes, another training
    # session is selected and replayed from its beginning.
    #
    # The current anomaly engine does not know about replay
    # boundaries, so longer sessions are preferred below.
    # =========================================================

    def _next_from_sessions(
        self,
        sessions,
        session_index,
        row_index,
        columns
    ):
        session = sessions[
            session_index
        ]

        row = session.iloc[
            row_index
        ]

        values = {
            col: float(row[col])
            for col in columns
        }

        row_index += 1

        if row_index >= len(session):

            row_index = 0

            session_index += 1

            if session_index >= len(sessions):
                session_index = 0

        return (
            values,
            session_index,
            row_index
        )

    # =========================================================
    # LOAD NEXT REAL-DERIVED NOMINAL VALUES
    # =========================================================

    def _load_next_nominal_values(self):

        (
            attitude,
            self.att_session_index,
            self.att_row_index
        ) = self._next_from_sessions(
            self.attitude_sessions,
            self.att_session_index,
            self.att_row_index,
            self.ATTITUDE_COLS
        )

        (
            power,
            self.power_session_index,
            self.power_row_index
        ) = self._next_from_sessions(
            self.power_sessions,
            self.power_session_index,
            self.power_row_index,
            self.POWER_COLS
        )

        (
            communication,
            self.comm_session_index,
            self.comm_row_index
        ) = self._next_from_sessions(
            self.communication_sessions,
            self.comm_session_index,
            self.comm_row_index,
            self.COMM_COLS
        )

        (
            thermal,
            self.thermal_session_index,
            self.thermal_row_index
        ) = self._next_from_sessions(
            self.thermal_sessions,
            self.thermal_session_index,
            self.thermal_row_index,
            self.THERMAL_COLS
        )

        # Attitude
        self.X_Coarse_Spin = (
            attitude["X_Coarse_Spin"]
        )

        self.Y_Coarse_Spin = (
            attitude["Y_Coarse_Spin"]
        )

        self.Z_Coarse_Spin = (
            attitude["Z_Coarse_Spin"]
        )

        # Power
        self.Battery_Voltage = (
            power["Battery_Voltage"]
        )

        self.Battery_Temp = (
            power["Battery_Temp"]
        )

        # Communication
        self.Background_RSSI = (
            communication["Background_RSSI"]
        )

        self.Last_RSSI = (
            communication["Last_RSSI"]
        )

        self.NanoCom_Temp = (
            communication["NanoCom_Temp"]
        )

        # Thermal
        self.HD_Cam_Temp = (
            thermal["HD_Cam_Temp"]
        )

        self.ACU2_Temp = (
            thermal["ACU2_Temp"]
        )

        self.NanoMind_Temp = (
            thermal["NanoMind_Temp"]
        )

        self.SDR_Temp = (
            thermal["SDR_Temp"]
        )

        self.SEPP_Temp = (
            thermal["SEPP_Temp"]
        )

    # =========================================================
    # FAULT CONTROL
    # =========================================================

    def inject_fault(
        self,
        fault,
        severity="moderate"
    ):
        self.active_faults[fault] = {
            "severity": severity,
            "start_step": self.step
        }

        print(
            f"FAULT STARTED -> "
            f"{fault} ({severity})"
        )

    def clear_fault(
        self,
        fault
    ):
        if fault in self.active_faults:

            del self.active_faults[fault]

            print(
                f"FAULT CLEARED -> {fault}"
            )

    def clear_all_faults(self):
        self.active_faults.clear()

    # =========================================================
    # CONTROLLED FAULT INJECTION
    # =========================================================

    def _apply_faults(self):

        severity_multiplier = {
            "mild": 1.0,
            "moderate": 2.0,
            "severe": 3.0
        }

        for fault, info in self.active_faults.items():

            m = severity_multiplier.get(
                info["severity"],
                2.0
            )

            age = (
                self.step
                - info["start_step"]
            )

            # =================================================
            # ATTITUDE
            # =================================================

            if fault == "spin_spike":

                if age % 4 == 0:
                    self.X_Coarse_Spin += (
                        0.004 * m
                    )

            elif fault == "spin_drift":

                drift = (
                    0.0007
                    * m
                    * (age + 1)
                )

                self.Y_Coarse_Spin += drift

            elif fault == "spin_oscillation":

                self.Z_Coarse_Spin += (
                    np.sin(age * 0.8)
                    * 0.003
                    * m
                )

            elif fault == "excessive_spin":

                offset = (
                    0.0012
                    * m
                    * (age + 1)
                )

                self.X_Coarse_Spin += offset
                self.Y_Coarse_Spin += offset
                self.Z_Coarse_Spin += offset

            # =================================================
            # POWER
            # =================================================

            elif fault == "voltage_drop":

                self.Battery_Voltage -= (
                    30.0 * m
                )

            elif fault == "voltage_degradation":

                self.Battery_Voltage -= (
                    10.0
                    * m
                    * (age + 1)
                )

            elif fault == "battery_heating":

                self.Battery_Temp += (
                    2.0
                    * m
                    * (age + 1)
                )

            elif fault == "coupled_power_stress":

                self.Battery_Voltage -= (
                    10.0
                    * m
                    * (age + 1)
                )

                self.Battery_Temp += (
                    1.5
                    * m
                    * (age + 1)
                )

            # =================================================
            # COMMUNICATION
            # =================================================

            elif (
                fault
                == "background_noise_elevation"
            ):

                self.Background_RSSI += (
                    2.0
                    * m
                    * (age + 1)
                )

            elif fault == "signal_degradation":

                if self.Last_RSSI != 0:

                    self.Last_RSSI -= (
                        2.0
                        * m
                        * (age + 1)
                    )

            elif (
                fault
                == "communication_heating"
            ):

                self.NanoCom_Temp += (
                    2.5
                    * m
                    * (age + 1)
                )

            elif (
                fault
                == "combined_link_stress"
            ):

                self.Background_RSSI += (
                    1.5
                    * m
                    * (age + 1)
                )

                if self.Last_RSSI != 0:

                    self.Last_RSSI -= (
                        1.5
                        * m
                        * (age + 1)
                    )

                self.NanoCom_Temp += (
                    1.5
                    * m
                    * (age + 1)
                )

            # =================================================
            # THERMAL
            # =================================================

            elif fault == "component_overheating":

                self.ACU2_Temp += (
                    3.0
                    * m
                    * (age + 1)
                )

            elif (
                fault
                == "gradual_thermal_drift"
            ):

                self.NanoMind_Temp += (
                    1.5
                    * m
                    * (age + 1)
                )

            elif fault == "thermal_divergence":

                self.ACU2_Temp += (
                    2.5
                    * m
                    * (age + 1)
                )

                self.NanoMind_Temp -= (
                    1.0
                    * m
                    * (age + 1)
                )

            elif (
                fault
                == "multi_component_heating"
            ):

                self.ACU2_Temp += (
                    2.0
                    * m
                    * (age + 1)
                )

                self.NanoMind_Temp += (
                    1.5
                    * m
                    * (age + 1)
                )

                self.SEPP_Temp += (
                    0.5
                    * m
                    * (age + 1)
                )

    # =========================================================
    # AGENT STATE
    # =========================================================

    def _update_battery_soc(self):

        drain = 0.002

        if (
            self.payload_subsystem
            == "ACTIVE_HIGH_DRAW"
        ):
            drain += 0.003

        voltage_deficit = max(
            0.0,
            7966.5
            - self.Battery_Voltage
        )

        temperature_excess = max(
            0.0,
            self.Battery_Temp
            - 256.0
        )

        drain += (
            voltage_deficit
            * 0.00002
        )

        drain += (
            temperature_excess
            * 0.00005
        )

        self.battery_soc = float(
            np.clip(
                self.battery_soc - drain,
                0.0,
                100.0
            )
        )

    def _derive_attitude_stability(self):

        # Project-derived state score.
        # It is not an ESA telemetry field.

        spin = np.array([
            self.X_Coarse_Spin,
            self.Y_Coarse_Spin,
            self.Z_Coarse_Spin
        ])

        magnitude = np.linalg.norm(
            spin
        )

        stability = (
            100.0
            - magnitude * 700.0
        )

        return round(
            float(
                np.clip(
                    stability,
                    0.0,
                    100.0
                )
            ),
            1
        )

    def _derive_communication_quality(self):

        # Last_RSSI == 0 is a sentinel/state in
        # OPS-SAT and must not be interpreted as 0 dBm.

        if self.Last_RSSI == 0:

            noise_penalty = max(
                0.0,
                self.Background_RSSI + 112.0
            )

            temperature_penalty = max(
                0.0,
                self.NanoCom_Temp - 350.0
            )

            quality = (
                70.0
                - 2.0 * noise_penalty
                - 0.4 * temperature_penalty
            )

        else:

            margin = (
                self.Last_RSSI
                - self.Background_RSSI
            )

            quality = (
                margin / 13.3
            ) * 100.0

            temperature_penalty = max(
                0.0,
                self.NanoCom_Temp - 350.0
            )

            quality -= (
                0.4
                * temperature_penalty
            )

        return round(
            float(
                np.clip(
                    quality,
                    0.0,
                    100.0
                )
            ),
            1
        )

    def _derive_thermal_margin(self):

        # Project simulation reference limits.
        # These are NOT claimed as ESA hardware limits.

        limits = {
            "ACU2": 420.0,
            "NanoMind": 360.0,
            "SEPP": 55.0
        }

        margins = [
            limits["ACU2"]
            - self.ACU2_Temp,

            limits["NanoMind"]
            - self.NanoMind_Temp,

            limits["SEPP"]
            - self.SEPP_Temp
        ]

        return round(
            float(min(margins)),
            1
        )

    def _update_operational_context(self):

        if (
            self.step > 0
            and self.step % 120 == 0
        ):
            if (
                self.payload_subsystem
                == "ACTIVE_HIGH_DRAW"
            ):
                self.payload_subsystem = (
                    "INACTIVE_STANDBY"
                )
            else:
                self.payload_subsystem = (
                    "ACTIVE_HIGH_DRAW"
                )

        if (
            self.step > 0
            and self.step % 5 == 0
        ):
            self.frc_horizon_minutes -= 1

        if self.frc_horizon_minutes <= 0:
            self.frc_horizon_minutes = 45

    # =========================================================
    # GENERATE PACKET
    # =========================================================

    def generate(self):

        # Get next real training-regime nominal state
        self._load_next_nominal_values()

        # Fault injection modifies raw telemetry only
        self._apply_faults()

        # Independent spacecraft state derivation
        self._update_battery_soc()

        self._update_operational_context()

        attitude_stability = (
            self._derive_attitude_stability()
        )

        communication_quality = (
            self._derive_communication_quality()
        )

        thermal_margin = (
            self._derive_thermal_margin()
        )

        packet = {

            "timestamp":
                datetime.now().isoformat(),

            # =================================================
            # EXACT SIX AGENT INPUTS
            # =================================================

            "agent_state": {

                "attitude_stability_percent":
                    attitude_stability,

                "communication_link_quality_percent":
                    communication_quality,

                "battery_soc_percent":
                    round(
                        self.battery_soc,
                        1
                    ),

                "thermal_margin":
                    thermal_margin,

                "payload_subsystem":
                    self.payload_subsystem,

                "frc_horizon_minutes":
                    self.frc_horizon_minutes
            },

            # =================================================
            # RAW ML INPUT
            # =================================================

            "raw_telemetry": {

                "attitude": {

                    "X_Coarse_Spin":
                        self.X_Coarse_Spin,

                    "Y_Coarse_Spin":
                        self.Y_Coarse_Spin,

                    "Z_Coarse_Spin":
                        self.Z_Coarse_Spin
                },

                "power": {

                    "Battery_Voltage":
                        self.Battery_Voltage,

                    "Battery_Temp":
                        self.Battery_Temp
                },

                "communication": {

                    "Background_RSSI":
                        self.Background_RSSI,

                    "Last_RSSI":
                        self.Last_RSSI,

                    "NanoCom_Temp":
                        self.NanoCom_Temp
                },

                "thermal": {

                    "HD_Cam_Temp":
                        self.HD_Cam_Temp,

                    "ACU2_Temp":
                        self.ACU2_Temp,

                    "NanoMind_Temp":
                        self.NanoMind_Temp,

                    "SDR_Temp":
                        self.SDR_Temp,

                    "SEPP_Temp":
                        self.SEPP_Temp
                }
            }
        }

        self.step += 1

        return packet

    # =========================================================
    # CONTINUOUS STREAM
    # =========================================================

    def stream(self):

        while True:

            yield self.generate()

            time.sleep(
                self.interval
            )


if __name__ == "__main__":

    simulator = (
        SatelliteTelemetrySimulator(
            csv_path="uhf_telemetry.csv",
            interval=0.5,
            seed=42
        )
    )

    try:

        for packet in simulator.stream():
            print(packet)

    except KeyboardInterrupt:

        print(
            "\nTelemetry stream stopped by user."
        )