from vehicle import Vehicle


class LLMVehicle(Vehicle):

    def __init__(self, config, id, vehicle_front=None):
        super().__init__(config, id, vehicle_front)
        self.last_llm_decision = None

    def update_acceleration(self):
        """
        LLM-based car-following model placeholder.

        Do not call an LLM at every simulation step. A practical version should
        use cached decisions, event-triggered decisions, or LLM-generated policy
        parameters, then convert those decisions into acceleration here.
        """
        raise NotImplementedError(
            "LLMVehicle needs a cached or event-triggered driving policy before "
            "it can compute acceleration."
        )

    def get_observation(self):
        """Return the current driving state that an LLM policy could inspect."""
        if self.vehicle_front is None:
            gap = None
            relative_speed = None
            front_speed = None
        else:
            gap = self.vehicle_front.position - self.position - self.vehicle_length
            relative_speed = self.speed - self.vehicle_front.speed
            front_speed = self.vehicle_front.speed

        return {
            "vehicle_id": self.id,
            "position": self.position,
            "speed": self.speed,
            "acceleration": self.a,
            "current_speed_limit": self.current_speed_limit,
            "gap": gap,
            "relative_speed": relative_speed,
            "front_speed": front_speed,
        }
