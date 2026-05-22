"""
Configuration loader: reads configs/optimize.yaml and produces flat CONFIG dict.
Explicit mapping allows tracing every key back to its YAML source.
Resolves map output path against project root for platform independence.
"""

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Project root: resolved relative to this file (src/config.py -> parents[1])."""
    return Path(__file__).resolve().parents[1]


def load_config(yaml_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load and flatten YAML config into flat dict with IRPConfig + DemandConfig keys.
    MAP_OUTPUT resolved to absolute path under project_root()/outputs.
    """
    if yaml_path is None:
        yaml_path = project_root() / "configs" / "optimize.yaml"
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {yaml_path}: {e}")

    cfg: dict[str, Any] = {
        # simulation
        "SIMULATION_DAYS":           y["simulation"]["days"],
        "PLANNING_HORIZON":          y["simulation"]["planning_horizon"],

        # data — real-demand toggle and CSV path
        "USE_REAL_DEMAND":           y["data"]["use_real_demand"],
        "REAL_DEMAND_CSV_PATH":      str(project_root() / y["data"]["real_demand_csv_path"]),
        "MISSING_DAY_EPS_MEAN":      y["data"]["missing_day_eps_mean"],
        "MISSING_DAY_EPS_SAFETY":    y["data"]["missing_day_eps_safety"],

        # fleet
        "NUM_VEHICLES":              y["fleet"]["num_vehicles"],
        "VEHICLE_CAPACITY":          y["fleet"]["vehicle_capacity"],
        "ATM_CAPACITY":              y["fleet"]["atm_capacity"],
        "INITIAL_INV_LOW":           y["fleet"]["initial_inventory_low_pct"],
        "INITIAL_INV_HIGH":          y["fleet"]["initial_inventory_high_pct"],
        "USE_HETEROGENEOUS_CAPACITY": y["fleet"]["use_heterogeneous_capacity"],

        # costs
        "HOLDING_COST_PER_DAY":      y["costs"]["holding_per_day"],
        "DISPATCH_COST_PER_VEHICLE": y["costs"]["dispatch_per_vehicle"],
        "DROP_FEE_PER_ATM":          y["costs"]["drop_fee_per_atm"],
        "TRAVEL_COST_PER_MIN":       y["costs"]["travel_per_min"],
        "MIN_LOAD_PER_VISIT":        y["costs"]["min_load_per_visit"],

        # service-level penalties and buffers
        "STOCKOUT_PENALTY":          y["service_level"]["stockout_penalty"],
        "SAFETY_FLOOR_PEN":          y["service_level"]["safety_floor_pen"],
        "EOH_FIXED_FEE":             y["service_level"]["eoh_fixed_fee"],
        "EOH_PEN_RATE":              y["service_level"]["eoh_pen_rate"],

        # service-time model
        "ONSITE_FIXED_MIN":          y["service_time"]["onsite_fixed_min"],
        "OFFSITE_FIXED_MIN":         y["service_time"]["offsite_fixed_min"],
        "CASSETTE_COEF":             y["service_time"]["cassette_coef_per_tl"],
        "SHIFT_LIMIT_MIN":           y["service_time"]["shift_limit_min"],

        # travel matrix
        "USE_OSM":                   y["travel_matrix"]["use_osm"],
        "OSM_PLACE":                 y["travel_matrix"]["osm_place"],
        "OSM_NETWORK_DIST_M":        y["travel_matrix"]["osm_network_dist_m"],
        "URBAN_SPEED_KMH":           y["travel_matrix"]["urban_speed_kmh"],
        "DETOUR_FACTOR":             y["travel_matrix"]["detour_factor"],
        # Symmetry flag: averages forward/backward OSM times if True (default).
        "SYMMETRIZE_TRAVEL_MATRIX":  y["travel_matrix"].get("symmetrize", True),

        # solver
        "MIP_GAP":                   y["solver"]["mip_gap"],
        "TIME_LIMIT_SEC":            y["solver"]["time_limit_sec"],
        "USE_SYMMETRY_BREAKING":     y["solver"]["use_symmetry_breaking"],

        # reproducibility
        "SEED":                      y["reproducibility"]["seed"],
    }

    # Path resolution: attach project-root-relative default so downstream sees absolute path.
    cfg["MAP_OUTPUT"] = str(
        project_root() / "outputs" / "istanbul_atm_simulation_map.html"
    )

    return cfg