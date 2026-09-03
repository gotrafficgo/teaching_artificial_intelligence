"""The main simulation loop."""

import random

from vehicle_idm import IDMVehicle


class Simulator:
    def __init__(self, config):
        self.config = config
        self.vehicles = []
        self.next_generation_time = 0.0

    def run(self, verbose=True):
        """Generate vehicles and update their states one time step at a time."""
        time_step = self.config.time_step
        number_of_steps = int(round(self.config.time_max / time_step))

        for step in range(number_of_steps):
            current_time = step * time_step

            if verbose and step > 0 and step % int(100 / time_step) == 0:
                print(f"step: {step}, time: {current_time:.0f}")

            # 1. Add a vehicle when its arrival time has been reached.
            self._generate_vehicle(current_time)

            active_vehicles = [vehicle for vehicle in self.vehicles if vehicle.active]

            # 2. Set the desired speed from the current road conditions.
            for vehicle in active_vehicles:
                vehicle.check_road(current_time)

            # 3. Calculate all accelerations before changing any vehicle state.
            for vehicle in active_vehicles:
                vehicle.update_acceleration()

            # 4. Update all speeds and positions.
            for vehicle in active_vehicles:
                vehicle.update_motion()

            # 5. Record only vehicles that are still on the road, plus the
            # final state of a vehicle that has just reached the road end.
            next_time = (step + 1) * time_step
            for vehicle in active_vehicles:
                vehicle.record_state(next_time)

        if verbose:
            inflow_rate = len(self.vehicles) / self.config.time_max * 3600
            print(f"\nVehicle number: {len(self.vehicles)}")
            print(f"Average admitted inflow: {inflow_rate:.0f} veh/h\n")

    def _generate_vehicle(self, current_time):
        """Generate at most one vehicle when the road entrance is clear."""
        if current_time + 1e-12 < self.next_generation_time:
            return

        active_vehicles = [vehicle for vehicle in self.vehicles if vehicle.active]
        vehicle_front = active_vehicles[-1] if active_vehicles else None

        # Do not insert a new vehicle on top of a queue at the entrance.
        if vehicle_front is not None:
            entry_gap = (
                self.config.vehicle_length
                + self.config.idm_minimum_spacing
                + self.config.initial_speed * self.config.idm_time_headway
            )
            if vehicle_front.position < entry_gap:
                return

        vehicle_id = len(self.vehicles) + 1
        vehicle = IDMVehicle(self.config, vehicle_id, vehicle_front)
        vehicle.record_state(current_time)
        self.vehicles.append(vehicle)

        interval = self.config.vehicle_min_interval
        if self.config.vehicle_extra_interval > 0:
            interval += random.expovariate(
                1.0 / self.config.vehicle_extra_interval
            )
        self.next_generation_time = self.next_generation_time + interval
