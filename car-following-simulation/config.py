"""All parameters for the car-following simulation."""

import random


class Config:
    """Keep the experiment settings in one easy-to-edit class."""

    DEFAULT_TIME_MAX = 1000.0

    def __init__(self, seed=1, experiment=1, time_max=DEFAULT_TIME_MAX):
        if experiment not in [1, 2, 3, 4]:
            raise ValueError("experiment must be 1, 2, 3, or 4")
        if time_max <= 0:
            raise ValueError("time_max must be positive")

        self.seed = seed
        random.seed(seed)

        # Road
        self.road_length = 5000.0             # m
        self.speed_limit = 30.0               # m/s

        # Simulation
        self.time_step = 0.1                  # s
        self.time_max = time_max              # s

        # IDM
        self.idm_minimum_spacing = 2.0        # s0, m
        self.idm_time_headway = 1.0           # T, s
        self.idm_maximum_acceleration = 1.5   # a, m/s^2
        self.idm_comfortable_deceleration = 2.0  # b, m/s^2
        self.idm_maximum_deceleration = 8.0   # numerical safety bound, m/s^2
        self.idm_acceleration_exponent = 4.0

        # Vehicle
        self.vehicle_length = 5.0             # m
        self.initial_speed = self.speed_limit
        self.relative_speed_noise = 0.5       # m/s

        # Temporary highway bottleneck
        self.bottleneck_x_start = self.road_length - 500.0      # m
        self.bottleneck_x_end   = self.road_length - 300.0        # m
        self.bottleneck_t_start = 100.0       # s
        self.bottleneck_speed_limit = 2.0     # m/s
        self.bottleneck_influence_rate = 0.7

        # Experiments 1 and 2 use a short bottleneck. Experiments 3 and 4
        # keep the bottleneck active until the simulation ends.
        if experiment in [1, 2]:
            self.bottleneck_t_end = 300.0
        else:
            self.bottleneck_t_end = self.time_max

        # Experiments 1 and 3 use fixed arrivals. Experiments 2 and 4 use
        # random arrivals with the same expected interval of 1.5 seconds.
        if experiment in [1, 3]:
            self.vehicle_min_interval = 1.7
            self.vehicle_extra_interval = 0.0
        else:
            self.vehicle_min_interval = 1.0
            self.vehicle_extra_interval = 0.5

        self.experiment = experiment
        self.reference_wave_speed_kmh = -16.0
