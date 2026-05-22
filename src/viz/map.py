"""
Folium map generator: renders static HTML map of depot + ATM-cluster topology.
Colour-coded by type (onsite/offsite). Takes explicit output_path for decoupled path resolution.
"""

from pathlib import Path

from ..data.spatial import SpatialData


try:
    import folium
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False


def generate_and_save_map(spatial: SpatialData, output_path: str | Path) -> None:
    """
    Render and save a Folium HTML map of the depot + ATM-cluster topology.

    Markers: depot in red (home icon), onsite clusters in blue, offsite ATMs
    in green (USD icon). Pop-ups show the location name and type.

    Silently no-ops with a warning if folium is unavailable (simulation continues).
    Output directory is created on demand. Save errors are caught and logged
    rather than raised because map output is not on the critical path.
    """
    if not FOLIUM_OK:
        print("[WARN] Folium library is missing. Map will not be generated.")
        return

    depot_lat, depot_lon = spatial["location_coords"][0]
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=12)

    folium.Marker(
        location=[depot_lat, depot_lon],
        popup="DEPOT: Cash Management Center",
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(m)

    for loc_id in range(1, spatial["num_locations"]):
        coords = spatial["location_coords"][loc_id]
        loc_type = spatial["location_type"][loc_id]
        loc_name = spatial["location_name"][loc_id]
        marker_color = "blue" if loc_type == "onsite" else "green"
        folium.Marker(
            location=coords,
            popup=f"{loc_name} ({loc_type})",
            icon=folium.Icon(color=marker_color, icon="usd"),
        ).add_to(m)

    try:
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out))
        print(f"[INFO] Map successfully generated and saved to: {out}")
    except Exception as e:
        print(f"[ERROR] Could not save map: {e}")