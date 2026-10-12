from dataclasses import dataclass, asdict, field
from typing import List, Dict


@dataclass
class RecoveryProcedure:
    procedure_id: str
    fault: str
    actions: List[str]
    expected_effects: Dict[str, float]
    preconditions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


PROCEDURE_LIBRARY = [

    # ---------------------------------------------------------
    # BATTERY
    # ---------------------------------------------------------

    RecoveryProcedure(
        "RP_01",
        "battery_low",
        ["disable_nonessential_payload"],
        {
            "battery_soc": +10.0
        },
        {}
    ),

    RecoveryProcedure(
        "RP_02",
        "battery_low",
        ["enter_safe_mode"],
        {
            "battery_soc": +15.0,
            "attitude_stability": -5.0
        },
        {}
    ),

    # ---------------------------------------------------------
    # ATTITUDE
    # ---------------------------------------------------------

    RecoveryProcedure(
        "RP_03",
        "attitude_fault",
        ["reset_attitude_control_computer"],
        {
            "attitude_stability": +40.0
        },
        {
            "battery_soc": 20.0
        }
    ),

    # NEW:
    # Alternative recovery procedure for attitude faults.
    # Used when controller reset becomes unsafe.
    RecoveryProcedure(
        "RP_07",
        "attitude_fault",
        ["enter_safe_mode"],
        {
            "battery_soc": +15.0,
            "attitude_stability": +5.0
        },
        {}
    ),

    # ---------------------------------------------------------
    # COMMUNICATION
    # ---------------------------------------------------------

    RecoveryProcedure(
        "RP_04",
        "comm_degradation",
        ["switch_to_backup_comm"],
        {
            "comm_link_quality": +50.0
        },
        {
            "battery_soc": 15.0
        }
    ),

    # ---------------------------------------------------------
    # THERMAL
    # ---------------------------------------------------------

    RecoveryProcedure(
        "RP_05",
        "thermal_excursion",
        ["activate_thermal_heater"],
        {
            "thermal_margin": +12.0
        },
        {
            "battery_soc": 10.0
        }
    ),

    # ---------------------------------------------------------
    # COMMUNICATION POWER MITIGATION
    # ---------------------------------------------------------

    RecoveryProcedure(
        "RP_06",
        "comm_degradation",
        ["reduce_transmit_power"],
        {
            "comm_link_quality": +20.0,
            "battery_soc": +5.0
        },
        {}
    ),
]


def check_preconditions(state, procedure: RecoveryProcedure) -> bool:

    state_dict = state.as_dict()

    for key, min_val in procedure.preconditions.items():

        if state_dict.get(key, 0) < min_val:
            return False

    return True