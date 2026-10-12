# spacecraft_state.py
from dataclasses import dataclass, asdict

@dataclass
class SpacecraftState:
    battery_soc: float
    thermal_margin: float
    attitude_stability: float
    comm_link_quality: float
    payload_active: bool
    timestamp: float

    def as_dict(self):
        return asdict(self)

def apply_fault(state: SpacecraftState, fault_type: str) -> SpacecraftState:
    s = SpacecraftState(**state.as_dict())
    if fault_type == "battery_low":
        s.battery_soc = max(0.0, s.battery_soc - 20.0)
    elif fault_type == "thermal_excursion":
        s.thermal_margin = max(0.0, s.thermal_margin - 8.0)
    elif fault_type == "attitude_fault":
        s.attitude_stability = max(0.0, s.attitude_stability - 30.0)
    elif fault_type == "comm_degradation":
        s.comm_link_quality = max(0.0, s.comm_link_quality - 40.0)
    return s