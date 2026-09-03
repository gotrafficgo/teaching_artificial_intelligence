from vehicle import Vehicle


class KNNVehicle(Vehicle):

    def __init__(self, config, id, vehicle_front=None):
        super().__init__(config, id, vehicle_front)
        self.k_neighbors = getattr(config, "knn_k_neighbors", None)
        self.training_data = None

    def update_acceleration(self):
        """
        KNN-based car-following model placeholder.

        A practical version should load or receive training samples, build a
        feature vector from the current driving state, find similar historical
        states, and convert their observed accelerations into this vehicle's
        acceleration.
        """
        raise NotImplementedError(
            "KNNVehicle needs training data and a neighbor-search policy before "
            "it can compute acceleration."
        )

    def get_features(self):
        """Return a numeric driving-state feature vector for a future KNN model."""
        if self.vehicle_front is None:
            gap = None
            relative_speed = None
            front_speed = None
        else:
            gap = self.vehicle_front.position - self.position - self.vehicle_length
            relative_speed = self.speed - self.vehicle_front.speed
            front_speed = self.vehicle_front.speed

        return {
            "speed": self.speed,
            "acceleration": self.a,
            "current_speed_limit": self.current_speed_limit,
            "gap": gap,
            "relative_speed": relative_speed,
            "front_speed": front_speed,
        }
