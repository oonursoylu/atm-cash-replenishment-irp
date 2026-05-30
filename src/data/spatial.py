"""
Spatial loader: hardcoded depot + 21 ATM-cluster topology (31 ATM IDs from real data).
"""

import io
from typing import TypedDict

import pandas as pd


# Hardcoded spatial topology
_LOCATIONS_CSV = """location_id,type,branch,name,lat,lon,num_atms,atms
0,depot,,Cash Management Center (Kozyatagi),40.982,29.105,0,
1,onsite,Z0241,"Kadikoy Merkez (on-site cluster, 3 ATMs)",40.9925,29.0277,3,"Z0241002,Z0241003,Z0241004"
2,offsite,Z0241,Marmara Uni Tip Fakultesi (Haydarpasa) [Z0241005],40.998,29.043,1,Z0241005
3,offsite,Z0241,Kadikoy IDO Iskelesi (Ferry) [Z0241006],40.991,29.0226,1,Z0241006
4,offsite,Z0241,Dr. Siyami Ersek Hastanesi [Z0241007],40.9877,29.0488,1,Z0241007
5,offsite,Z0241,Suadiye Bagdat Cad. [Z0241008],40.958,29.07,1,Z0241008
6,onsite,Z0276,"Bostanci (on-site cluster, 2 ATMs)",40.953,29.095,2,"Z0276001,Z0276002"
7,offsite,Z0276,Bostanci Minibus/Otogar Duragi [Z0276003],40.947,29.0963,1,Z0276003
8,onsite,Z0386,"Umraniye Merkez (on-site cluster, 3 ATMs)",41.0156,29.1245,3,"Z0386001,Z0386002,Z0386003"
9,offsite,Z0386,Umraniye Egitim Arastirma Hst. [Z0386004],41.0293,29.1195,1,Z0386004
10,offsite,Z0386,Umraniye Belediyesi [Z0386005],41.0172,29.1105,1,Z0386005
11,offsite,Z0386,Inkilap Mah. Carsi [Z0386006],41.01,29.115,1,Z0386006
12,offsite,Z0386,Dudullu Sanayi Sitesi [Z0386007],41.023,29.157,1,Z0386007
13,onsite,Z0646,"Goztepe (on-site cluster, 2 ATMs)",40.97,29.067,2,"Z0646001,Z0646002"
14,offsite,Z0646,Fenerbahce Stadyumu [Z0646003],40.9878,29.0368,1,Z0646003
15,offsite,Z0646,Goztepe 60. Yil Parki [Z0646004],40.975,29.062,1,Z0646004
16,onsite,Z0951,"Uskudar (on-site cluster, 2 ATMs)",41.0227,29.015,2,"Z0951001,Z0951002"
17,onsite,Z1031,"Erenkoy (on-site cluster, 2 ATMs)",40.965,29.085,2,"Z1031001,Z1031002"
18,onsite,Z1119,"Acibadem (on-site cluster, 1 ATMs)",40.993,29.049,1,Z1119001
19,onsite,Z1524,"Moda (on-site cluster, 1 ATMs)",40.983,29.034,1,Z1524001
20,onsite,Z1899,"Sahrayicedit (on-site cluster, 2 ATMs)",40.9715,29.0947,2,"Z1899001,Z1899002"
21,onsite,Z2135,"Atasehir (on-site cluster, 2 ATMs)",40.992,29.128,2,"Z2135001,Z2135002"
"""


# Tertile assignment from train-period active-day mean withdrawal.
# Capacities ~1.3x train max-day per tier, rounded to 50K.
ATM_TIERS: dict[str, str] = {
    'Z0646003': 'low',  'Z0646004': 'low',  'Z0386005': 'low',
    'Z0386006': 'low',  'Z0241008': 'low',  'Z0386004': 'low',
    'Z0276003': 'low',  'Z0241006': 'low',  'Z0241005': 'low',
    'Z0386007': 'low',  'Z0241007': 'low',
    'Z1031001': 'mid',  'Z0276002': 'mid',  'Z1031002': 'mid',
    'Z1899002': 'mid',  'Z1899001': 'mid',  'Z0386003': 'mid',
    'Z1524001': 'mid',  'Z0386001': 'mid',  'Z0276001': 'mid',
    'Z0386002': 'mid',
    'Z0241003': 'high', 'Z0241004': 'high', 'Z2135002': 'high',
    'Z0951002': 'high', 'Z0646001': 'high', 'Z0646002': 'high',
    'Z0951001': 'high', 'Z2135001': 'high', 'Z1119001': 'high',
    'Z0241002': 'high',
}

TIER_CAPACITY: dict[str, int] = {
    'low':  250_000,
    'mid':  400_000,
    'high': 500_000,
}


class SpatialData(TypedDict):
    """Container for spatial information. All consumers index via these keys."""
    atm_location: dict[str, int]
    location_atms: dict[int, list[str]]
    location_coords: dict[int, tuple[float, float]]
    location_type: dict[int, str]
    location_branch: dict[int, str | None]
    location_name: dict[int, str]
    num_locations: int


def load_hardcoded_spatial() -> SpatialData:
    """Parse embedded LOCATIONS_CSV into SpatialData container (deterministic, no I/O)."""
    df = pd.read_csv(io.StringIO(_LOCATIONS_CSV))
    df["branch"] = df["branch"].where(df["branch"].notna(), None)
    df["atms"] = df["atms"].where(df["atms"].notna(), "")

    spatial: SpatialData = {
        "atm_location":    {},
        "location_atms":   {},
        "location_coords": {},
        "location_type":   {},
        "location_branch": {},
        "location_name":   {},
        "num_locations":   0,
    }

    for _, row in df.iterrows():
        l = int(row["location_id"])
        spatial["location_coords"][l] = (float(row["lat"]), float(row["lon"]))
        spatial["location_type"][l]   = str(row["type"])
        spatial["location_branch"][l] = row["branch"] if row["branch"] else None
        spatial["location_name"][l]   = str(row["name"])

        atm_cell = str(row["atms"]).strip()
        atms_here = [a.strip() for a in atm_cell.split(",") if a.strip()] if atm_cell else []
        spatial["location_atms"][l] = atms_here
        for a in atms_here:
            spatial["atm_location"][a] = l

    spatial["num_locations"] = len(spatial["location_coords"])
    return spatial


def get_capacity_per_atm() -> dict[str, float]:
    """Per-ATM capacity from tier assignment (heterogeneous mode).
    Caller is responsible for invoking only when use_heterogeneous_capacity=True."""
    return {a: float(TIER_CAPACITY[t]) for a, t in ATM_TIERS.items()}