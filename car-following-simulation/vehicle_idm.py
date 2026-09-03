"""The Intelligent Driver Model acceleration rule."""

import random

from vehicle import Vehicle


class IDMVehicle(Vehicle):
    def __init__(self, config, vehicle_id, vehicle_front=None):
        super().__init__(config, vehicle_id, vehicle_front)

        self.minimum_spacing = config.idm_minimum_spacing
        self.time_headway = config.idm_time_headway
        self.comfortable_deceleration = config.idm_comfortable_deceleration
        self.acceleration_exponent = config.idm_acceleration_exponent
        self.relative_speed_noise = config.relative_speed_noise

    def update_acceleration(self):
        """Calculate IDM acceleration from the current traffic state."""
        desired_speed = self.current_speed_limit

        # Free-road term: accelerate toward the desired speed.
        free_road_term = (self.speed / desired_speed) ** self.acceleration_exponent

        # Interaction term: brake when approaching a slower or close leader.
        interaction_term = 0.0
        leader = self.vehicle_front
        if leader is not None and leader.active:
            gap = leader.position - self.position - self.vehicle_length
            gap = max(gap, 0.1)

            relative_speed = self.speed - leader.speed
            perceived_relative_speed = relative_speed + random.gauss(
                0.0, self.relative_speed_noise
            )

            braking_term = (
                self.speed
                * perceived_relative_speed
                / (
                    2
                    * (
                        self.maximum_acceleration
                        * self.comfortable_deceleration
                    )
                    ** 0.5
                )
            )
            desired_gap = self.minimum_spacing + max(
                0.0,
                self.speed * self.time_headway + braking_term,
            )
            interaction_term = (desired_gap / gap) ** 2

        self.acceleration = self.maximum_acceleration * (
            1.0 - free_road_term - interaction_term
        )
        self.acceleration = max(
            -self.maximum_deceleration,
            self.acceleration,
        )
