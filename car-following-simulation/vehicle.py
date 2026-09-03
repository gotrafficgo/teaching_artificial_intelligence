"""State and motion shared by every vehicle model."""

import random


class Vehicle:
    def __init__(self, config, vehicle_id, vehicle_front=None):
        self.id = vehicle_id
        self.position = 0.0
        self.speed = config.initial_speed
        self.acceleration = 0.0
        self.vehicle_front = vehicle_front
        self.active = True
        self.history = []

        self.road_length = config.road_length
        self.speed_limit = config.speed_limit
        self.current_speed_limit = self.speed_limit
        self.vehicle_length = config.vehicle_length
        self.time_step = config.time_step
        self.maximum_acceleration = config.idm_maximum_acceleration
        self.maximum_deceleration = config.idm_maximum_deceleration

        self.bottleneck_x_start = config.bottleneck_x_start
        self.bottleneck_x_end = config.bottleneck_x_end
        self.bottleneck_t_start = config.bottleneck_t_start
        self.bottleneck_t_end = config.bottleneck_t_end
        self.bottleneck_speed_limit = config.bottleneck_speed_limit
        self.bottleneck_entry_time = None
        self.was_inside_bottleneck = False
        self.influenced_by_bottleneck = (
            random.random() < config.bottleneck_influence_rate
        )

    def check_road(self, current_time):
        """Choose the desired speed for the current road and time."""
        inside_bottleneck = (
            self.bottleneck_x_start
            <= self.position
            <= self.bottleneck_x_end
        )
        bottleneck_is_active = (
            self.bottleneck_t_start
            <= current_time
            < self.bottleneck_t_end
        )

        # Record the first time this vehicle enters the bottleneck area.
        if inside_bottleneck and not self.was_inside_bottleneck:
            self.bottleneck_entry_time = current_time
        self.was_inside_bottleneck = inside_bottleneck

        entered_after_bottleneck_started = (
            self.bottleneck_entry_time is not None
            and self.bottleneck_t_start
            <= self.bottleneck_entry_time
            < self.bottleneck_t_end
        )

        if (
            self.influenced_by_bottleneck
            and inside_bottleneck
            and bottleneck_is_active
            and entered_after_bottleneck_started
        ):
            self.current_speed_limit = self.bottleneck_speed_limit
        else:
            self.current_speed_limit = self.speed_limit

    def update_acceleration(self):
        """Calculate acceleration in a subclass such as IDMVehicle."""
        raise NotImplementedError

    def update_motion(self):
        """Update speed and position together using average speed."""
        old_speed = self.speed

        # Keep acceleration inside the allowed physical range.
        acceleration = max(
            -self.maximum_deceleration,
            min(self.acceleration, self.maximum_acceleration),
        )

        # The vehicle cannot move backwards.
        new_speed = max(0.0, old_speed + acceleration * self.time_step)

        # Average-speed integration keeps speed and position consistent.
        distance = 0.5 * (old_speed + new_speed) * self.time_step
        self.position = min(self.road_length, self.position + distance)
        self.speed = new_speed

        # If speed was clipped at zero, store the acceleration actually applied.
        self.acceleration = (new_speed - old_speed) / self.time_step

        if self.position >= self.road_length:
            self.active = False

    def record_state(self, current_time):
        self.history.append(
            {
                "t": current_time,
                "position": self.position,
                "speed": self.speed,
                "acceleration": self.acceleration,
            }
        )
