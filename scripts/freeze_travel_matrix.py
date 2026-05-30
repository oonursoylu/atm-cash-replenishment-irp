"""
Freeze the travel-time matrix for reproducible runs.

Builds the depot-centred OSM drive network ONCE and derives BOTH the symmetric
and the asymmetric travel matrix from that single download, then serialises each
to data/frozen/:

    data/frozen/travel_matrix_symmetric.json
    data/frozen/travel_matrix_asymmetric.json

Each artifact carries: location order, tau, a content hash, tau sum, and build
metadata (backend, symmetrize flag, timestamp, osmnx/networkx/python versions,
depot centroid, network radius, urban speed, detour factor, speed bounds).

Once frozen, build_travel_matrix() prefers these files automatically (code-level
default; no config change). Re-run this script to regenerate. This script does
NOT decide which matrix is canonical -- it emits artifacts and prints hashes.

Usage (from project root):
    python scripts/freeze_travel_matrix.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.spatial import load_hardcoded_spatial
from src.data.travel import freeze_travel_matrices


def main() -> int:
    cfg = load_config()
    if not cfg["USE_OSM"]:
        print("[WARN] cfg USE_OSM is false; freezing OSM matrices anyway "
              "(haversine is symmetric by construction and needs no freeze).")

    spatial = load_hardcoded_spatial()
    print("=" * 82)
    print("FREEZING TRAVEL MATRICES (single OSM download -> symmetric + asymmetric)")
    print("=" * 82)
    summary = freeze_travel_matrices(spatial, cfg)

    print("\n" + "=" * 82)
    print("FROZEN ARTIFACTS")
    print("=" * 82)
    for mode in ("symmetric", "asymmetric"):
        s = summary[mode]
        print(f"  {mode:>10}: hash={s['content_hash']}  tau_sum={s['tau_sum']}")
        print(f"  {'':>10}  {s['path']}")
    print("\nActive default is the SYMMETRIC artifact (cfg symmetrize=true).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
