"""Empty extension point for a future LLM vehicle policy."""

from vehicle import Vehicle


class LLMVehicle(Vehicle):
    def update_acceleration(self):
        raise NotImplementedError(
            "Add a cached or event-triggered policy before using LLMVehicle."
        )

    def get_observation(self):
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
            "vehicle_id": self.id,
            "position": self.position,
            "speed": self.speed,
            "acceleration": self.acceleration,
            "current_speed_limit": self.current_speed_limit,
            "gap": gap,
            "relative_speed": relative_speed,
            "front_speed": front_speed,
        }
