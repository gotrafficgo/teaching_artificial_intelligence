import argparse
from pathlib import Path

from config import Config
from simulator import Simulator


def main():
    parser = argparse.ArgumentParser(description="Run a car-following simulation.")
    parser.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Experiment scenario: 1-4 (default: 1)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for saving the time-space diagram.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive plot window.",
    )
    args = parser.parse_args()

    # Load simulation parameters
    config = Config(experiment=args.experiment)

    # Initialize simulator
    sim = Simulator(config)

    # Run the simulation loop
    sim.run()

    # Generate the time–space diagram
    from plotting import plot_time_space_diagram
    plot_time_space_diagram(
        sim,
        config,
        output=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
