import random

class Config:
    def __init__(self, seed=1, experiment=1):

        # === Initialize Random Seeds ===
        self.seed = seed
        random.seed(self.seed)

        # === Road Configuration ===
        self.speed_limit = 30                # Speed limit (m/s)
        self.road_length = 2000              # Total road length (m)

        # === Simulation Settings ===
        self.simulation_time_step = 0.1      # Simulation time step Δt (s)
        self.time_max             = 1000     # Total simulation time (s)

        # === IDM Parameters ===
        self.idm_minimum_spacing      = 2    # Minimum spacing s0 (m)
        self.idm_safety_time_headway  = 1    # Safety time headway T (s)
        self.idm_acceleration         = 1.5  # Maximum acceleration a (m/s²)
        self.idm_desired_deceleration = 2    # Comfortable deceleration b (m/s²)
        self.idm_delay                = 0.4  # Reaction delay τ (s)

        self.vehicle_length       = 5                 # Vehicle length L (m)
        self.initial_speed        = self.speed_limit  # Initial speed (m/s)
        self.initial_acceleration = 0                 # Initial acceleration (m/s²)
        self.relative_speed_noise = 0.5               # Noise σ for perceived speed (m/s)

        # === Bottleneck Settings ===
        self.bottleneck_length = 200                         # Bottleneck length (m)
        self.bottleneck_x_start = self.road_length - 500     # Bottleneck start position (m)
        self.bottleneck_x_end = self.bottleneck_x_start + self.bottleneck_length  # End position (m)
        self.bottleneck_t_start = 100                        # Activation time (s)
        self.bottleneck_speed_limit = self.speed_limit * 0.2 # Reduced speed limit (m/s)
        self.percentage_influenced_by_bottleneck = 0.7       # Fraction of vehicles affected



        # === Experiment Selection ===
        # 1: Deterministic inflow + short bottleneck
        # 2: Stochastic inflow + short bottleneck
        # 3: Deterministic inflow + long bottleneck
        # 4: Stochastic inflow + long bottleneck
        self.experiment = experiment

        # === Vehicle Generation Settings ===
        # Vehicle inter-arrival time = min_interval + exponential(extra_interval)
        if self.experiment == 1:
            self.vehicle_min_interval = 2.5
            self.vehicle_extra_interval = 0
            self.bottleneck_t_end = 200

        elif self.experiment == 2:
            self.vehicle_min_interval = 1.5
            self.vehicle_extra_interval = 1
            self.bottleneck_t_end = 200

        elif self.experiment == 3:
            self.vehicle_min_interval = 2.5
            self.vehicle_extra_interval = 0
            self.bottleneck_t_end = self.time_max

        elif self.experiment == 4:
            self.vehicle_min_interval = 1.5
            self.vehicle_extra_interval = 1
            self.bottleneck_t_end = self.time_max
