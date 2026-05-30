"""
Travel-time matrix: OSM (real network) or Haversine (fallback) backend.
Symmetry handling via SYMMETRIZE_TRAVEL_MATRIX flag (averages OSM forward/backward times).
"""

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from .spatial import SpatialData


try:
    import osmnx as ox
    import networkx as nx
    OSMNX_OK = True
except ImportError:
    OSMNX_OK = False


# (i, j) -> minutes. Diagonal is 0.
TravelMatrix = dict[tuple[int, int], float]
TravelBackend = str  # "OSM" | "HAVERSINE"


class TravelConfig(TypedDict):
    """CONFIG keys consumed by this module. Other keys are ignored."""
    USE_OSM: bool
    OSM_PLACE: str
    OSM_NETWORK_DIST_M: int
    URBAN_SPEED_KMH: float
    DETOUR_FACTOR: float
    SYMMETRIZE_TRAVEL_MATRIX: bool


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres. Pure function."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def travel_matrix_haversine(
    spatial: SpatialData, cfg: TravelConfig
) -> tuple[TravelMatrix, TravelBackend]:
    """Symmetric travel-time matrix from haversine + detour factor, rounded to 2 decimals."""
    coords = spatial["location_coords"]
    nL = spatial["num_locations"]
    tt: TravelMatrix = {}
    for i in range(nL):
        lat1, lon1 = coords[i]
        for j in range(nL):
            if i == j:
                tt[(i, j)] = 0.0
            else:
                lat2, lon2 = coords[j]
                d_km = haversine_km(lat1, lon1, lat2, lon2) * cfg["DETOUR_FACTOR"]
                tt[(i, j)] = round(d_km / cfg["URBAN_SPEED_KMH"] * 60.0, 2)
    return tt, "HAVERSINE"


# Per-edge speed bound (km/h) applied to OSM 'maxspeed' tags. Module constant so
# both live builds and the freeze artifact metadata reference the same numbers.
_OSM_SPEED_BOUNDS_KMH = (5.0, 80.0)


def _download_weighted_graph(spatial: SpatialData, cfg: TravelConfig):
    """
    Download the depot-centred OSM drive network ONCE and annotate every edge
    with 'travel_time_min'. Pure side-effect-free w.r.t. cfg; returns the graph.

    Factored out of travel_matrix_osm so the freeze step can derive BOTH the
    symmetric and the asymmetric matrix from a single download (clean ablation).

    Speed normalisation: OSM 'maxspeed' tags are dirty (strings, lists, missing),
    so per-edge speed is bounded to [5, 80] km/h to neutralise mis-tagged
    motorways and pedestrian segments.
    """
    if not OSMNX_OK:
        raise RuntimeError("osmnx not installed")

    depot_lat, depot_lon = spatial["location_coords"][0]
    print(f"[OSM] Downloading drive network around depot ({depot_lat:.4f}, {depot_lon:.4f})...")
    G = ox.graph_from_point(
        (depot_lat, depot_lon),
        dist=cfg["OSM_NETWORK_DIST_M"],
        network_type="drive",
    )

    lo, hi = _OSM_SPEED_BOUNDS_KMH
    for _u, _v, _k, d in G.edges(keys=True, data=True):
        length_m = d.get("length", 0)
        speed_kmh = d.get("maxspeed", cfg["URBAN_SPEED_KMH"])
        try:
            speed_kmh = float(speed_kmh) if not isinstance(speed_kmh, list) else float(speed_kmh[0])
        except (ValueError, TypeError, IndexError):
            speed_kmh = cfg["URBAN_SPEED_KMH"]
        speed_kmh = max(lo, min(speed_kmh, hi))
        d["travel_time_min"] = (length_m / 1000.0) / speed_kmh * 60.0

    return G


def _osm_asymmetric_matrix(spatial: SpatialData, G) -> TravelMatrix:
    """
    Raw asymmetric Dijkstra travel times on the weighted graph G, with the
    disconnected-node fallback applied (independent of symmetrization): when a
    Dijkstra result is infinite the reciprocal direction is used; if both are
    infinite the sentinel 999.0 is assigned.
    """
    nL = spatial["num_locations"]
    nearest = {
        l: ox.distance.nearest_nodes(
            G, spatial["location_coords"][l][1], spatial["location_coords"][l][0]
        )
        for l in range(nL)
    }

    tt: TravelMatrix = {}
    for i in range(nL):
        lengths = nx.single_source_dijkstra_path_length(G, nearest[i], weight="travel_time_min")
        for j in range(nL):
            tt[(i, j)] = round(lengths.get(nearest[j], float("inf")), 2) if i != j else 0.0

    # Disconnected-node fallback: applied regardless of symmetrization flag.
    for i in range(nL):
        for j in range(i + 1, nL):
            a, b = tt[(i, j)], tt[(j, i)]
            if math.isinf(a) and math.isinf(b):
                tt[(i, j)] = tt[(j, i)] = 999.0
            elif math.isinf(a):
                tt[(i, j)] = b
            elif math.isinf(b):
                tt[(j, i)] = a

    return tt


def _symmetrize_matrix(tt: TravelMatrix, nL: int) -> TravelMatrix:
    """Average forward/backward times into a symmetric matrix (returns a new dict)."""
    out = dict(tt)
    for i in range(nL):
        for j in range(i + 1, nL):
            avg = (out[(i, j)] + out[(j, i)]) / 2.0
            out[(i, j)] = out[(j, i)] = round(avg, 2)
    return out


def travel_matrix_osm(
    spatial: SpatialData, cfg: TravelConfig
) -> tuple[TravelMatrix, TravelBackend]:
    """
    Travel-time matrix from OSM drive network shortest paths.

    Behaviour is unchanged from the pre-refactor version: download + speed
    normalisation, asymmetric Dijkstra with disconnected-node fallback, then
    optional symmetrization (cfg["SYMMETRIZE_TRAVEL_MATRIX"], default True;
    averages forward/backward OSM times).

    Raises RuntimeError if osmnx is not installed.
    """
    G = _download_weighted_graph(spatial, cfg)
    tt = _osm_asymmetric_matrix(spatial, G)
    if cfg.get("SYMMETRIZE_TRAVEL_MATRIX", True):
        tt = _symmetrize_matrix(tt, spatial["num_locations"])
    return tt, "OSM"


# ---- frozen-matrix artifacts ----------------------------------------------
#
# Reproducibility motivation: live build_travel_matrix() yields slightly
# different OSM matrices across separate Python processes (osmnx/networkx graph
# construction is not bit-stable run-to-run). Sweeps run each cell in its own
# subprocess, so cells could silently disagree on tau. Freezing one artifact and
# loading it everywhere removes that source of cross-process drift.
#
# These are CODE-LEVEL defaults: no new config key. If the frozen file exists it
# is preferred; otherwise build_travel_matrix() emits a visible warning and falls
# back to the live OSM/haversine path (never silently).

_FROZEN_SCHEMA_VERSION = 1
_FROZEN_SYMMETRIC_NAME = "travel_matrix_symmetric.json"
_FROZEN_ASYMMETRIC_NAME = "travel_matrix_asymmetric.json"


def _project_root() -> Path:
    """Project root: src/data/travel.py -> parents[2]."""
    return Path(__file__).resolve().parents[2]


def frozen_dir() -> Path:
    """Directory holding frozen travel-matrix artifacts (data/frozen)."""
    return _project_root() / "data" / "frozen"


def frozen_matrix_path(symmetrize: bool) -> Path:
    """Path to the frozen artifact for the requested symmetrization mode."""
    name = _FROZEN_SYMMETRIC_NAME if symmetrize else _FROZEN_ASYMMETRIC_NAME
    return frozen_dir() / name


def matrix_content_hash(tt: TravelMatrix) -> str:
    """
    Canonical content hash of a travel matrix (md5 over sorted, 6-dp-rounded
    (i, j, tau) triples). Identical formula to the reproducibility probe so the
    frozen artifact's hash is directly comparable to probe output.
    """
    items = sorted((int(i), int(j), round(float(v), 6)) for (i, j), v in tt.items())
    return hashlib.md5(repr(items).encode()).hexdigest()


def _serialize_matrix(
    tt: TravelMatrix,
    spatial: SpatialData,
    cfg: TravelConfig,
    symmetrize: bool,
    backend: TravelBackend,
) -> dict:
    """Build the JSON-serialisable frozen-artifact payload for one matrix."""
    nL = spatial["num_locations"]
    location_order = list(range(nL))
    depot_lat, depot_lon = spatial["location_coords"][0]
    try:
        import networkx as _nx
        nx_version = _nx.__version__
    except Exception:
        nx_version = None
    return {
        "schema_version": _FROZEN_SCHEMA_VERSION,
        "backend": backend,
        "symmetrize": symmetrize,
        "num_locations": nL,
        "location_order": location_order,
        "content_hash": matrix_content_hash(tt),
        "tau_sum": round(sum(tt.values()), 2),
        "metadata": {
            "build_timestamp": datetime.now(timezone.utc).isoformat(),
            "osmnx_version": (ox.__version__ if OSMNX_OK else None),
            "networkx_version": nx_version,
            "python_version": sys.version.split()[0],
            "depot_centroid": [depot_lat, depot_lon],
            "osm_network_dist_m": cfg["OSM_NETWORK_DIST_M"],
            "urban_speed_kmh": cfg["URBAN_SPEED_KMH"],
            "detour_factor": cfg["DETOUR_FACTOR"],
            "speed_bounds_kmh": list(_OSM_SPEED_BOUNDS_KMH),
            "note": "symmetric and asymmetric matrices both derived from a single OSM download",
        },
        # tau keyed by "i,j" strings (JSON has no tuple keys).
        "tau": {f"{i},{j}": v for (i, j), v in tt.items()},
    }


def freeze_travel_matrices(
    spatial: SpatialData, cfg: TravelConfig, out_dir: Path | None = None
) -> dict:
    """
    Build the base OSM graph ONCE and derive BOTH the symmetric and asymmetric
    travel matrices from that single download, then serialise each to
    data/frozen/. Returns a summary {mode: {path, content_hash, tau_sum}}.

    Does NOT decide which matrix is canonical: it emits artifacts and reports
    their hashes. Swapping in a different matrix later is a single file write.
    """
    out_dir = out_dir or frozen_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    G = _download_weighted_graph(spatial, cfg)
    asym = _osm_asymmetric_matrix(spatial, G)
    sym = _symmetrize_matrix(asym, spatial["num_locations"])

    summary: dict = {}
    for symmetrize, tt in ((True, sym), (False, asym)):
        payload = _serialize_matrix(tt, spatial, cfg, symmetrize, "OSM")
        path = out_dir / (_FROZEN_SYMMETRIC_NAME if symmetrize else _FROZEN_ASYMMETRIC_NAME)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary["symmetric" if symmetrize else "asymmetric"] = {
            "path": str(path),
            "content_hash": payload["content_hash"],
            "tau_sum": payload["tau_sum"],
        }
    return summary


def load_frozen_matrix(
    spatial: SpatialData, symmetrize: bool
) -> tuple[TravelMatrix, TravelBackend] | None:
    """
    Load the frozen travel matrix for the requested symmetrization mode.

    Returns (tt, backend) on success, or None if the file is absent or fails a
    consistency check (so the caller can fall back to a live build). A stored
    content hash that no longer matches the deserialised tau emits a warning and
    is treated as a load failure.
    """
    path = frozen_matrix_path(symmetrize)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        nL = spatial["num_locations"]
        if payload.get("num_locations") != nL:
            print(f"[WARN] Frozen matrix {path.name} has num_locations="
                  f"{payload.get('num_locations')} != spatial {nL}; ignoring frozen file.")
            return None
        tt: TravelMatrix = {}
        for key, v in payload["tau"].items():
            i_str, j_str = key.split(",")
            tt[(int(i_str), int(j_str))] = float(v)
        recomputed = matrix_content_hash(tt)
        if payload.get("content_hash") != recomputed:
            print(f"[WARN] Frozen matrix {path.name} content_hash mismatch "
                  f"(stored {payload.get('content_hash')}, recomputed {recomputed}); "
                  "ignoring frozen file.")
            return None
        backend: TravelBackend = payload.get("backend", "OSM")
        print(f"[FROZEN] Loaded {path.name} backend={backend} "
              f"hash={recomputed} tau_sum={payload.get('tau_sum')}")
        return tt, backend
    except Exception as e:
        print(f"[WARN] Failed to read frozen matrix {path.name}: {e}; ignoring frozen file.")
        return None


def build_travel_matrix(
    spatial: SpatialData, cfg: TravelConfig
) -> tuple[TravelMatrix, TravelBackend]:
    """
    Top-level entry. Reproducibility-preferring order:

      1. If USE_OSM and a frozen artifact exists for the requested symmetrize
         mode, load it (cross-process-stable; works even without osmnx).
      2. Else if USE_OSM and osmnx is available, live-build from OSM after a
         VISIBLE warning that the result is not cross-process reproducible.
      3. Else fall back to haversine (symmetric by construction).

    The live OSM path is retained, never silently bypassed.
    """
    if cfg["USE_OSM"]:
        symmetrize = cfg.get("SYMMETRIZE_TRAVEL_MATRIX", True)
        frozen = load_frozen_matrix(spatial, symmetrize)
        if frozen is not None:
            return frozen
        print(f"[WARN] No frozen travel matrix at {frozen_matrix_path(symmetrize)} "
              "-- falling back to LIVE build (NOT reproducible across processes). "
              "Run scripts/freeze_travel_matrix.py to create the frozen artifact.")
        if OSMNX_OK:
            try:
                return travel_matrix_osm(spatial, cfg)
            except Exception as e:
                print(f"[WARN] OSM failed: {e}")
    return travel_matrix_haversine(spatial, cfg)