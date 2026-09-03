# Car-Following Simulation

This project demonstrates how individual vehicle behavior can produce
large-scale traffic patterns. Vehicles use the Intelligent Driver Model (IDM)
to select acceleration, while a temporary low-speed road segment creates
queues and backward-moving congestion waves.

## Learning objectives

After completing the project, you should be able to:

- distinguish a vehicle behavior model from a simulation environment;
- explain how gap, relative speed, and desired speed affect IDM acceleration;
- compare deterministic and stochastic traffic inflows;
- interpret a speed-colored time-space diagram.

## Project files

| File | Purpose |
| --- | --- |
| `config.py` | Defines the road, vehicle, IDM, bottleneck, and experiment parameters |
| `vehicle.py` | Defines shared vehicle state and motion updates |
| `vehicle_idm.py` | Implements the IDM acceleration rule |
| `simulator.py` | Generates vehicles and advances the simulation |
| `plotting.py` | Creates the time-space diagram |
| `vehicle_knn.py` | Provides an unfinished extension point for a data-driven policy |
| `vehicle_llm.py` | Provides an unfinished extension point for an LLM-assisted policy |

The KNN and LLM classes are exercises and are not used by the default
simulation.

## Run an experiment

From the repository root, activate the shared environment and run:

```bash
python car-following-simulation/main.py --experiment 1
```

To save the figure without opening a window:

```bash
python car-following-simulation/main.py --experiment 1 \
  --output car-following-simulation/results/experiment-1.png --no-show
```

The four experiments combine two inflow processes and two bottleneck
durations:

| Experiment | Vehicle inflow | Bottleneck duration |
| --- | --- | --- |
| 1 | Fixed interval | Short, from 100 to 200 seconds |
| 2 | Random interval | Short, from 100 to 200 seconds |
| 3 | Fixed interval | From 100 seconds to the end |
| 4 | Random interval | From 100 seconds to the end |

The trajectory color represents vehicle speed. The white dashed line is a
reference congestion-wave speed of `-16 km/h`.

## IDM relationship

The model combines a free-driving term with a following-distance term:

```text
desired gap = minimum gap + speed x time headway + relative-speed correction
acceleration = maximum acceleration
               x [1 - (speed / desired speed)^4 - (desired gap / gap)^2]
```

The simulation also constrains acceleration, prevents negative speed, and
applies a minimum numerical gap.

## Suggested experiments

- Compare experiments 1 and 2 to isolate the effect of inflow randomness.
- Compare experiments 1 and 3 to isolate the effect of bottleneck duration.
- Change the time headway and observe how congestion formation changes.
- Change the random seed and test whether the conclusions remain consistent.
