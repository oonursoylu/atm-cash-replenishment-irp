# Toy-data baseline for the IRP optimisation layer (synthetic demand).
# Single-depot multi-vehicle inventory routing on a 7-day rolling horizon,
# solved with CPLEX MILP. Preserved as the reference baseline against which
# the toy-data Model Parameter Justification was calibrated.
# The production pipeline lives in src/.

import os
import io
import math
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from docplex.mp.model import Model

try:
    import osmnx as ox
    import networkx as nx
    OSMNX_OK = True
except ImportError:
    OSMNX_OK = False

try:
    import folium
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False


# CONFIG
CONFIG = {
    # Simulation
    "SIMULATION_DAYS": 30,
    "PLANNING_HORIZON": 7,

    # Fleet & physical limits
    "NUM_VEHICLES": 2,
    "VEHICLE_CAPACITY": 1_500_000,
    "ATM_CAPACITY": 400_000,
    "INITIAL_INV_LOW": 0.40,
    "INITIAL_INV_HIGH": 0.60,

    # Costs
    "HOLDING_COST_PER_DAY": 0.0005,
    "DISPATCH_COST_PER_VEHICLE": 300,
    "DROP_FEE_PER_ATM": 50,
    "TRAVEL_COST_PER_MIN": 0.5,
    "MIN_LOAD_PER_VISIT": 20_000,

    # Service-level penalties
    "STOCKOUT_PENALTY": 5000,
    "EOH_FIXED_FEE": 300,
    "EOH_PEN_RATE": 0.005,
    "SAFETY_BUFFER_MULT": 1.30,
    "SAFETY_FLOOR_PEN": 0.1,

    # Service time
    "ONSITE_FIXED_MIN": 20,
    "OFFSITE_FIXED_MIN": 25,
    "CASSETTE_COEF": 1.0 / 20_000,
    "SHIFT_LIMIT_MIN": 480,

    # OSM travel-time matrix
    "USE_OSM": True,
    "OSM_PLACE": "Kadikoy, Istanbul, Turkey",
    "OSM_NETWORK_DIST_M": 15_000,
    "URBAN_SPEED_KMH": 20,
    "DETOUR_FACTOR": 1.35,

    # Solver
    "MIP_GAP": 0.05,
    "TIME_LIMIT_SEC": 600,
    "USE_SYMMETRY_BREAKING": True,

    # Output
    "MAP_OUTPUT": (
        os.path.join(os.path.expanduser("~/Desktop"), "istanbul_atm_simulation_map.html")
        if os.path.isdir(os.path.expanduser("~/Desktop"))
        else os.path.abspath("istanbul_atm_simulation_map.html")
    ),

    "SEED": 42,
}

random.seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])


# SECTION 1: SPATIAL STRUCTURE (Hardcoded)
LOCATIONS_CSV = """location_id,type,branch,name,lat,lon,num_atms,atms
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

def load_hardcoded_spatial():
    df = pd.read_csv(io.StringIO(LOCATIONS_CSV))
    df["branch"] = df["branch"].where(df["branch"].notna(), None)
    df["atms"] = df["atms"].where(df["atms"].notna(), "")

    spatial = {
        "atm_location":    {},
        "location_atms":   {},
        "location_coords": {},
        "location_type":   {},
        "location_branch": {},
        "location_name":   {},
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


# SECTION 2: TRAVEL TIME MATRIX
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def travel_matrix_haversine(spatial, cfg):
    coords = spatial["location_coords"]
    nL = spatial["num_locations"]
    tt = {}
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

def travel_matrix_osm(spatial, cfg):
    if not OSMNX_OK:
        raise RuntimeError("osmnx not installed")

    depot_lat, depot_lon = spatial["location_coords"][0]
    print(f"[OSM] Downloading drive network around depot ({depot_lat:.4f}, {depot_lon:.4f})...")
    G = ox.graph_from_point((depot_lat, depot_lon), dist=cfg["OSM_NETWORK_DIST_M"], network_type='drive')

    for u, v, k, d in G.edges(keys=True, data=True):
        length_m = d.get("length", 0)
        speed_kmh = d.get("maxspeed", cfg["URBAN_SPEED_KMH"])
        try:
            speed_kmh = float(speed_kmh) if not isinstance(speed_kmh, list) else float(speed_kmh[0])
        except (ValueError, TypeError, IndexError):
            speed_kmh = cfg["URBAN_SPEED_KMH"]
        speed_kmh = max(5.0, min(speed_kmh, 80.0))
        d["travel_time_min"] = (length_m / 1000.0) / speed_kmh * 60.0

    nL = spatial["num_locations"]
    nearest = {l: ox.distance.nearest_nodes(G, spatial["location_coords"][l][1], spatial["location_coords"][l][0]) for l in range(nL)}

    tt = {}
    for i in range(nL):
        lengths = nx.single_source_dijkstra_path_length(G, nearest[i], weight="travel_time_min")
        for j in range(nL):
            tt[(i, j)] = round(lengths.get(nearest[j], float("inf")), 2) if i != j else 0.0

    for i in range(nL):
        for j in range(i + 1, nL):
            a = tt[(i, j)]; b = tt[(j, i)]
            if math.isinf(a): a = b
            if math.isinf(b): b = a
            avg = (a + b) / 2 if not math.isinf(a) else 999.0
            tt[(i, j)] = tt[(j, i)] = round(avg, 2)

    return tt, "OSM"

def build_travel_matrix(spatial, cfg):
    if cfg["USE_OSM"] and OSMNX_OK:
        try:
            return travel_matrix_osm(spatial, cfg)
        except Exception as e:
            print(f"[WARN] OSM failed: {e}")
    return travel_matrix_haversine(spatial, cfg)


# SECTION 3: MASTER TIMESERIES GENERATOR
_ATM_ARCHETYPES = {
    "low":  {"mean_range": (2_500, 16_000),  "cv_range": (0.80, 1.30)},
    "mid":  {"mean_range": (18_000, 55_000), "cv_range": (0.70, 1.10)},
    "high": {"mean_range": (55_000, 98_000), "cv_range": (0.60, 0.95)},
}
_DOW_FACTORS = [1.29, 1.09, 1.04, 1.05, 1.22, 0.84, 0.47]

def _assign_archetype(atm_id: str) -> str:
    h = sum(ord(c) for c in atm_id)
    if h % 3 == 0: return "low"
    elif h % 3 == 1: return "mid"
    return "high"

def generate_master_timeseries(atms, cfg):
    total_days = cfg["SIMULATION_DAYS"] + cfg["PLANNING_HORIZON"]
    C = cfg["ATM_CAPACITY"]
    rng_np = np.random.default_rng(cfg["SEED"])
    rng_py = random.Random(cfg["SEED"])

    d_mean, d_safety, actual_demand = {}, {}, {}
    I0 = {}

    for a in sorted(atms):
        archetype = _assign_archetype(a)
        base_lo, base_hi = _ATM_ARCHETYPES[archetype]["mean_range"]
        cv_lo, cv_hi = _ATM_ARCHETYPES[archetype]["cv_range"]

        base_mean = rng_py.uniform(base_lo, base_hi)
        cv = rng_py.uniform(cv_lo, cv_hi)
        safety_mult = rng_py.uniform(1.80, 4.00)

        for t in range(1, total_days + 1):
            dow = (t - 1) % 7
            dow_factor = _DOW_FACTORS[dow]
            mu_t = base_mean * dow_factor

            forecast_noise = rng_np.normal(loc=1.0, scale=0.15)
            mean_val = max(0.0, mu_t * forecast_noise)
            jitter = rng_np.uniform(0.95, 1.08)
            safety_val = max(mean_val, mean_val * safety_mult * jitter)

            d50_cap = 60_000
            d90_cap = 130_000
            mean_val = min(mean_val, d50_cap)
            safety_val = min(safety_val, d90_cap)
            safety_val = max(safety_val, mean_val)

            actual_noise = rng_np.normal(loc=1.0, scale=0.20)
            actual_val = max(0.0, mu_t * actual_noise)
            actual_val = min(actual_val, C)

            d_mean[(a, t)] = mean_val
            d_safety[(a, t)] = safety_val
            actual_demand[(a, t)] = actual_val

        I0[a] = rng_py.uniform(cfg["INITIAL_INV_LOW"], cfg["INITIAL_INV_HIGH"]) * C

    return d_mean, d_safety, actual_demand, I0


# SECTION 4: ROLLING HORIZON OPTIMIZER
def solve_single_horizon(sim_day, actual_inventory, master_data, cfg):
    T = list(range(1, cfg["PLANNING_HORIZON"] + 1))
    K = list(range(cfg["NUM_VEHICLES"]))
    sp = master_data["spatial"]
    L = list(range(sp["num_locations"]))
    I = sorted(sp["atm_location"].keys())

    buf = cfg["SAFETY_BUFFER_MULT"]

    # Physical depletion (no buffer): used in inventory balance constraint.
    # The buffer multiplier enters separately as a soft floor on remaining
    # inventory (Bertsimas & Sim 2004 robust IRP formulation), so the balance
    # equation tracks expected physical cash flow rather than a buffered proxy.
    d_phys = {(a, t): min(master_data["d_mean"][(a, sim_day + t - 1)], cfg["ATM_CAPACITY"]) for a in I for t in T}
    d_safe = {(a, t): min(master_data["d_safety"][(a, sim_day + t - 1)] * buf, cfg["ATM_CAPACITY"]) for a in I for t in T}

    eoh = {a: float(np.mean([d_safe[(a, t)] for t in T])) for a in I}

    depot = 0
    L_visit = [l for l in L if l != depot]
    N = len(L_visit)
    onsite_locs = [l for l in L_visit if sp["location_type"][l] == "onsite"]
    offsite_locs = [l for l in L_visit if sp["location_type"][l] == "offsite"]

    mdl = Model(name=f"IRP_Day_{sim_day}")

    x = mdl.binary_var_dict([(i, j, k, t) for i in L for j in L if i != j for k in K for t in T], name="x")
    y = mdl.binary_var_dict([(l, k, t) for l in L_visit for k in K for t in T], name="y")
    z = mdl.binary_var_dict([(a, k, t) for a in I for k in K for t in T], name="z")
    w = mdl.binary_var_dict([(k, t) for k in K for t in T], name="w")
    q = mdl.continuous_var_dict([(a, k, t) for a in I for k in K for t in T], lb=0, name="q")
    Inv = mdl.continuous_var_dict([(a, t) for a in I for t in T], lb=0, ub=cfg["ATM_CAPACITY"], name="Inv")
    s = mdl.continuous_var_dict([(a, t) for a in I for t in T], lb=0, name="s")
    safety_slack = mdl.continuous_var_dict([(a, t) for a in I for t in T], lb=0, name="safety_slack")
    eoh_slack = mdl.continuous_var_dict(I, lb=0, name="eoh_slack")
    is_stockout = mdl.binary_var_dict([(a, t) for a in I for t in T], name="is_stockout")
    is_eoh_short = mdl.binary_var_dict(I, name="is_eoh_short")
    u = mdl.continuous_var_dict([(l, k, t) for l in L_visit for k in K for t in T], lb=1, ub=N, name="u")

    tt = master_data["travel_time"]
    travel = mdl.sum(tt[(i, j)] * cfg["TRAVEL_COST_PER_MIN"] * x[(i, j, k, t)] for i in L for j in L if i != j for k in K for t in T)
    dispatch = mdl.sum(cfg["DISPATCH_COST_PER_VEHICLE"] * w[(k, t)] for k in K for t in T)
    drops = mdl.sum(cfg["DROP_FEE_PER_ATM"] * z[(a, k, t)] for a in I for k in K for t in T)
    hold = mdl.sum(cfg["HOLDING_COST_PER_DAY"] * Inv[(a, t)] for a in I for t in T)
    stock = mdl.sum(cfg["STOCKOUT_PENALTY"] * is_stockout[(a, t)] for a in I for t in T)
    # EOH penalty: hybrid (fixed activation fee + per-TL slack rate).
    # Fixed fee deters complete neglect of an ATM at horizon end; slack rate
    # ensures the optimiser minimises the magnitude of any shortfall.
    eoh_cost = mdl.sum(cfg["EOH_FIXED_FEE"] * is_eoh_short[a] + cfg["EOH_PEN_RATE"] * eoh_slack[a] for a in I)
    safety_cost = mdl.sum(cfg["SAFETY_FLOOR_PEN"] * safety_slack[(a, t)] for a in I for t in T)
    mdl.minimize(travel + dispatch + drops + hold + stock + eoh_cost + safety_cost)

    for k in K:
        for t in T:
            for l in L_visit:
                mdl.add_constraint(mdl.sum(x[(i, l, k, t)] for i in L if i != l) == y[(l, k, t)])
                mdl.add_constraint(mdl.sum(x[(l, j, k, t)] for j in L if j != l) == y[(l, k, t)])
            out_flow = mdl.sum(x[(depot, j, k, t)] for j in L if j != depot)
            in_flow = mdl.sum(x[(i, depot, k, t)] for i in L if i != depot)
            mdl.add_constraint(out_flow == w[(k, t)])
            mdl.add_constraint(in_flow == w[(k, t)])

    for a in I:
        l = sp["atm_location"][a]
        for k in K:
            for t in T:
                mdl.add_constraint(z[(a, k, t)] <= y[(l, k, t)])
                mdl.add_constraint(q[(a, k, t)] <= cfg["ATM_CAPACITY"] * z[(a, k, t)])
                mdl.add_constraint(q[(a, k, t)] >= cfg["MIN_LOAD_PER_VISIT"] * z[(a, k, t)])

    for l in L_visit:
        for t in T:
            mdl.add_constraint(mdl.sum(y[(l, k, t)] for k in K) <= 1)
            for k in K:
                mdl.add_constraint(y[(l, k, t)] <= w[(k, t)])

    for a in I:
        for t in T:
            mdl.add_constraint(mdl.sum(z[(a, k, t)] for k in K) <= 1)
            prev = actual_inventory[a] if t == T[0] else Inv[(a, t - 1)]
            tot_in = mdl.sum(q[(a, k, t)] for k in K)
            # Physical inventory balance (no buffer in deduction)
            mdl.add_constraint(Inv[(a, t)] == prev + tot_in - d_phys[(a, t)] + s[(a, t)])
            mdl.add_constraint(prev + tot_in <= cfg["ATM_CAPACITY"])
            mdl.add_constraint(s[(a, t)] <= d_phys[(a, t)] * is_stockout[(a, t)])
            # Safety floor: remaining inventory should cover (buf - 1) x demand
            mdl.add_constraint(Inv[(a, t)] + safety_slack[(a, t)] >= (buf - 1.0) * d_phys[(a, t)])

    T_end = T[-1]
    for a in I:
        mdl.add_constraint(Inv[(a, T_end)] + eoh_slack[a] >= eoh[a])
        mdl.add_constraint(eoh_slack[a] <= eoh[a] * is_eoh_short[a])

    for k in K:
        for t in T:
            mdl.add_constraint(mdl.sum(q[(a, k, t)] for a in I) <= cfg["VEHICLE_CAPACITY"])
            travel_expr = mdl.sum(tt[(i, j)] * x[(i, j, k, t)] for i in L for j in L if i != j)
            onsite_fixed = mdl.sum(cfg["ONSITE_FIXED_MIN"] * y[(l, k, t)] for l in onsite_locs)
            offsite_fixed = mdl.sum(cfg["OFFSITE_FIXED_MIN"] * y[(l, k, t)] for l in offsite_locs)
            cassette = mdl.sum(cfg["CASSETTE_COEF"] * q[(a, k, t)] for a in I)
            mdl.add_constraint(travel_expr + onsite_fixed + offsite_fixed + cassette <= cfg["SHIFT_LIMIT_MIN"])

            for l in L_visit:
                for m in L_visit:
                    if l != m:
                        mdl.add_constraint(u[(l, k, t)] - u[(m, k, t)] + N * x[(l, m, k, t)] <= N - 1)

    if cfg["USE_SYMMETRY_BREAKING"]:
        for t in T:
            for k in range(len(K) - 1):
                mdl.add_constraint(w[(K[k], t)] >= w[(K[k + 1], t)])

    mdl.parameters.mip.tolerances.mipgap = cfg["MIP_GAP"]
    mdl.parameters.timelimit = cfg["TIME_LIMIT_SEC"]
    mdl.parameters.emphasis.mip = 1
    mdl.parameters.mip.strategy.heuristicfreq = 20
    mdl.parameters.mip.strategy.rinsheur = 50
    mdl.parameters.mip.strategy.probe = 2

    sol = mdl.solve(log_output=False)

    # Initialise q with zeros for every ATM so downstream code can index
    # actions["q"][a] safely even if the solve fails or returns no incumbent.
    day1_actions = {"q": {a: 0.0 for a in I}, "routes": {}}

    # Verify a feasible integer solution exists before reading variables.
    has_solution = (sol is not None) and (mdl.solve_details.status_code in (101, 102, 105, 107, 113))

    if has_solution:
        for a in I:
            day1_actions["q"][a] = sum(q[(a, k, 1)].solution_value for k in K)

        # Use 0.9 threshold (instead of 0.5) for binary variables; under a
        # 5% MIP gap, integer values of 0.51 or 0.99 can occur and the
        # tighter threshold avoids ambiguous route reconstruction.
        for k in K:
            used = any(x[(i, j, k, 1)].solution_value > 0.9 for i in L for j in L if i != j)
            if used:
                route = [depot]
                cur = depot
                seen = set()
                # Cap iterations at len(L) to guarantee termination even if
                # the LP relaxation produces a malformed cycle.
                for _ in range(len(L)):
                    nxt = None
                    for j in L:
                        if j != cur and x[(cur, j, k, 1)].solution_value > 0.9:
                            nxt = j
                            break
                    if nxt is None or nxt in seen:
                        break
                    if nxt == depot:
                        route.append(depot)
                        break
                    route.append(nxt)
                    seen.add(nxt)
                    cur = nxt
                day1_actions["routes"][k] = route

    # Release CPLEX C-API resources held by the model. Without this, repeated
    # daily calls in the rolling horizon accumulate memory across the run.
    mdl.end()

    return day1_actions, (sol if has_solution else None)


# SECTION 5: MAP GENERATION
def generate_and_save_map(spatial, cfg):
    # Function to generate and save the Folium map based on current spatial data
    if not FOLIUM_OK:
        print("[WARN] Folium library is missing. Map will not be generated.")
        return

    depot_lat, depot_lon = spatial["location_coords"][0]
    
    # Initialize the base map centered around the depot
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12)

    # Add the depot marker with a distinct color and icon
    folium.Marker(
        location=[depot_lat, depot_lon],
        popup="DEPOT: Cash Management Center",
        icon=folium.Icon(color="red", icon="home")
    ).add_to(m)

    # Add markers for all ATM locations
    for loc_id in range(1, spatial["num_locations"]):
        coords = spatial["location_coords"][loc_id]
        loc_type = spatial["location_type"][loc_id]
        loc_name = spatial["location_name"][loc_id]
        
        # Color coding: blue for onsite, green for offsite
        marker_color = "blue" if loc_type == "onsite" else "green"
        
        folium.Marker(
            location=coords,
            popup=f"{loc_name} ({loc_type})",
            icon=folium.Icon(color=marker_color, icon="usd")
        ).add_to(m)

    # Export the generated map to the specified HTML path
    try:
        out_path = cfg["MAP_OUTPUT"]
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        m.save(out_path)
        print(f"[INFO] Map successfully generated and saved to: {out_path}")
    except Exception as e:
        print(f"[ERROR] Could not save map: {e}")


# SECTION 6: SIMULATION EXECUTION LOOP
def run_simulation(cfg):
    print("=" * 82)
    print(f"STARTING {cfg['SIMULATION_DAYS']}-DAY ROLLING HORIZON SIMULATION")
    print(f"USING OPTIMAL PARAMS: {cfg['STOCKOUT_PENALTY']} TL Penalty & {cfg['SAFETY_BUFFER_MULT']}x Buffer")
    print("=" * 82)
    start_time = time.time()

    sp = load_hardcoded_spatial()
    tt, _ = build_travel_matrix(sp, cfg)
    atms = sorted(sp["atm_location"].keys())

    d_mean, d_safe, actual_demand, I0 = generate_master_timeseries(atms, cfg)
    master_data = {"spatial": sp, "travel_time": tt, "d_mean": d_mean, "d_safety": d_safe}

    # Generate the HTML map before starting the simulation loop
    generate_and_save_map(sp, cfg)

    actual_inventory = I0.copy()

    sim_kpis = {
        "travel_cost": 0, "dispatch_cost": 0, "drop_fees": 0,
        "holding_cost": 0, "stockout_cost": 0, "stockout_events": 0,
        "total_deliveries": 0, "total_dispatches": 0
    }

    for sim_day in range(1, cfg["SIMULATION_DAYS"] + 1):
        print(f"\n[DAY {sim_day}/{cfg['SIMULATION_DAYS']}] Solving 7-day horizon...")

        actions, sol = solve_single_horizon(sim_day, actual_inventory, master_data, cfg)

        if sol is None:
            print(f"  [!] INFEASIBLE on Day {sim_day}. Stopping simulation.")
            break

        daily_dispatches = len(actions["routes"])
        daily_stops = sum(1 for a, qty in actions["q"].items() if qty > 0)
        daily_delivery_tl = sum(actions["q"].values())

        sim_kpis["total_dispatches"] += daily_dispatches
        sim_kpis["dispatch_cost"] += daily_dispatches * cfg["DISPATCH_COST_PER_VEHICLE"]
        sim_kpis["drop_fees"] += daily_stops * cfg["DROP_FEE_PER_ATM"]
        sim_kpis["total_deliveries"] += daily_delivery_tl

        day_travel_cost = 0
        for r in actions["routes"].values():
            day_travel_cost += sum(tt[(r[i], r[i+1])] * cfg["TRAVEL_COST_PER_MIN"] for i in range(len(r)-1))
        sim_kpis["travel_cost"] += day_travel_cost

        daily_stockout_events = 0
        daily_holding_tl = 0

        for a in atms:
            delivered = actions["q"][a]
            demanded = actual_demand[(a, sim_day)]

            new_inv = actual_inventory[a] + delivered - demanded

            if new_inv < 0:
                daily_stockout_events += 1
                sim_kpis["stockout_events"] += 1
                sim_kpis["stockout_cost"] += cfg["STOCKOUT_PENALTY"]
                new_inv = 0

            new_inv = min(new_inv, cfg["ATM_CAPACITY"])
            actual_inventory[a] = new_inv
            daily_holding_tl += new_inv * cfg["HOLDING_COST_PER_DAY"]

        sim_kpis["holding_cost"] += daily_holding_tl

        print(f"  -> Executed: {daily_dispatches} vehicles, {daily_stops} stops, {daily_delivery_tl:,.0f} TL loaded.")
        print(f"  -> Reality:  {daily_stockout_events} stockout events occurred today.")

    total_time = time.time() - start_time

    print("\n" + "=" * 82)
    print(f"SIMULATION COMPLETE: {cfg['SIMULATION_DAYS']} DAYS EXECUTED")
    print("=" * 82)
    print(f"Total Compute Time : {total_time:.1f} seconds ({total_time/60:.1f} mins)")

    total_cost = (sim_kpis["travel_cost"] + sim_kpis["dispatch_cost"] +
                  sim_kpis["drop_fees"] + sim_kpis["holding_cost"] + sim_kpis["stockout_cost"])

    print("\nOPERATIONAL KPIs (Actuals)")
    print(f"  Total Cost        : {total_cost:,.2f} TL")
    print(f"  Total Dispatches  : {sim_kpis['total_dispatches']} vehicle shifts")
    print(f"  Total Cash Loaded : {sim_kpis['total_deliveries']:,.0f} TL")
    print(f"  Stockout Events   : {sim_kpis['stockout_events']} events (over {cfg['SIMULATION_DAYS'] * len(atms)} ATM-days)")

    print("\nCOST BREAKDOWN")
    print(f"  Travel Cost       : {sim_kpis['travel_cost']:>12,.2f} TL")
    print(f"  Dispatch Cost     : {sim_kpis['dispatch_cost']:>12,.2f} TL")
    print(f"  Drop Fees         : {sim_kpis['drop_fees']:>12,.2f} TL")
    print(f"  Holding Cost      : {sim_kpis['holding_cost']:>12,.2f} TL")
    print(f"  Stockout Penalties: {sim_kpis['stockout_cost']:>12,.2f} TL")

if __name__ == "__main__":
    run_simulation(CONFIG)