import random


class Vehicle:

    def __init__(self, config, id, vehicle_front=None):
        self.id = id
        self.position = 0
        self.vehicle_front = vehicle_front

        # Road and vehicle initialization
        self.road_length = config.road_length
        self.speed_limit = config.speed_limit
        self.current_speed_limit = self.speed_limit
        self.speed = config.initial_speed
        self.a = config.initial_acceleration
        self.vehicle_length = config.vehicle_length
        self.delta_t = config.simulation_time_step
        self.history = []

        # Bottleneck parameters
        self.bottleneck_x_start = config.bottleneck_x_start
        self.bottleneck_x_end = config.bottleneck_x_end
        self.bottleneck_t_start = config.bottleneck_t_start
        self.bottleneck_t_end = config.bottleneck_t_end
        self.bottleneck_speed_limit = config.bottleneck_speed_limit

        # Whether this vehicle reacts to bottleneck limits
        self.influenced_by_bottleneck = (
            random.random() < config.percentage_influenced_by_bottleneck
        )

    def check_road(self, current_time):
        """Apply bottleneck speed limit if vehicle is inside spatial and temporal bottleneck."""
        if not self.influenced_by_bottleneck:
            self.current_speed_limit = self.speed_limit
            return

        in_x_range = (self.position >= self.bottleneck_x_start and
                      self.position <= self.bottleneck_x_end)
        in_t_range = (current_time >= self.bottleneck_t_start and
                      current_time <= self.bottleneck_t_end)

        if in_x_range and in_t_range:
            self.current_speed_limit = self.bottleneck_speed_limit
        else:
            self.current_speed_limit = self.speed_limit

    def update_acceleration(self):
        """Compute vehicle acceleration in a concrete car-following model."""
        raise NotImplementedError("Subclasses must implement update_acceleration().")

    def update_speed(self):
        """Update vehicle speed using acceleration and safety constraints."""

        # Vehicle has left the road
        if self.position >= self.road_length:
            v_new = self.speed_limit
        else:
            a = self.a
            v = self.speed
            delta_t = self.delta_t

            # Standard Euler update
            v_new = v + a * delta_t

            # Additional constraint: do not exceed max speed allowed by gap
            if self.vehicle_front is not None:
                s = self.vehicle_front.position - self.position - self.vehicle_length
                s = max(s, 0.01)  # [additional constraint]
                v_max_allowed = s / delta_t
                v_new = min(v_new, v_max_allowed)

            # Prevent negative speeds
            v_new = max(v_new, 0)  # [additional constraint]

        self.speed = v_new

    def update_position(self):
        """Update vehicle position with kinematic equation and constraints."""
        v = self.speed
        a = self.a
        delta_t = self.delta_t

        # d = v*dt + 0.5*a*dt^2
        d = v * delta_t + 0.5 * a * delta_t ** 2
        d = max(d, 0)  # [additional constraint]

        self.position = self.position + d

    def record_state(self, t):
        """Store vehicle state for later analysis."""
        self.history.append({
            "t": t,
            "position": self.position,
            "speed": self.speed,
            "acceleration": self.a
        })
