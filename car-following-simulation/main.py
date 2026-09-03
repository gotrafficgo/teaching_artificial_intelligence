"""Run one car-following experiment."""

import argparse
from pathlib import Path

from config import Config
from plotting import plot_time_space_diagram
from simulator import Simulator


def main():
    parser = argparse.ArgumentParser(description="Run a car-following simulation.")
    parser.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
    )
    parser.add_argument(
        "--time-max",
        type=float,
        default=Config.DEFAULT_TIME_MAX,
        help="Simulation duration in seconds.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    config = Config(
        experiment=args.experiment,
        time_max=args.time_max,
    )
    simulator = Simulator(config)
    simulator.run()
    plot_time_space_diagram(
        simulator,
        config,
        output=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
