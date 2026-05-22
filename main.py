"""
Project entry point: loads YAML config, wires map-generation callback, runs rolling-horizon simulation.

Usage: python main.py
"""

from src.config import load_config
from src.sim.rolling_horizon import run_simulation
from src.viz.map import generate_and_save_map


def main() -> None:
    cfg = load_config()

    # Bridge the (spatial, output_path) signature of generate_and_save_map
    # to the (spatial, cfg) callback shape that run_simulation expects.
    # The path lives in cfg["MAP_OUTPUT"], already resolved by the loader.
    def _map_fn(spatial, cfg_inner):
        generate_and_save_map(spatial, cfg_inner["MAP_OUTPUT"])

    run_simulation(cfg, map_generator=_map_fn)


if __name__ == "__main__":
    main()