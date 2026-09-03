"""Empty extension point for a future KNN vehicle model."""

from vehicle import Vehicle


class KNNVehicle(Vehicle):
    def update_acceleration(self):
        raise NotImplementedError(
            "Add training data and a KNN policy before using KNNVehicle."
        )

    def get_features(self):
        leader = self.vehicle_front
        if leader is None or not leader.active:
            gap = None
            relative_speed = None
            front_speed = None
        else:
            gap = leader.position - self.position - self.vehicle_length
            relative_speed = self.speed - leader.speed
            front_speed = leader.speed

        return {
            "speed": self.speed,
            "acceleration": self.acceleration,
            "current_speed_limit": self.current_speed_limit,
            "gap": gap,
            "relative_speed": relative_speed,
            "front_speed": front_speed,
        }
