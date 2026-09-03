import random

from vehicle import Vehicle


class IDMVehicle(Vehicle):

    def __init__(self, config, id, vehicle_front=None):
        super().__init__(config, id, vehicle_front)

        # IDM parameters
        self.s0 = config.idm_minimum_spacing
        self.T = config.idm_safety_time_headway
        self.a_max = config.idm_acceleration
        self.b_desired = config.idm_desired_deceleration
        self.tau = config.idm_delay

        # Noise in relative speed perception
        self.relative_speed_noise = config.relative_speed_noise

    def update_acceleration(self):
        """Compute IDM acceleration with additional constraints."""

        # Handle case with no front vehicle
        if self.vehicle_front is None:
            v_front_speed = self.speed_limit
            v_front_position = self.position + 1e6  # effectively infinite headway
        else:
            v_front_speed = self.vehicle_front.speed
            v_front_position = self.vehicle_front.position

        v = self.speed
        v_delta = v - v_front_speed  # relative speed
        v_delta_perceived = self._perceptive_relative_speed(v_delta)

        # Net distance gap
        s = v_front_position - self.position - self.vehicle_length
        s = max(s, 0.1)  # [additional constraint]

        # IDM parameters
        s0 = self.s0
        a_max = self.a_max
        b_desired = self.b_desired

        # Desired dynamical gap s*
        term1 = self.T * v
        term2 = v * v_delta_perceived / (2 * (a_max * b_desired) ** 0.5)
        s_star = s0 + max(0, term1 + term2)

        # IDM acceleration formula
        desired_speed = self.current_speed_limit
        term1 = (v / desired_speed) ** 4
        term2 = (s_star / s) ** 2
        a = a_max * (1 - term1 - term2)

        # [additional constraint]
        if a < -b_desired:
            a = -b_desired
        if a > a_max:
            a = a_max

        self.a = a

    def _perceptive_relative_speed(self, v_delta):
        """Add noise to perceived relative speed."""
        noise = random.gauss(0, self.relative_speed_noise)
        return v_delta + noise
