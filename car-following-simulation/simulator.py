import random
from config import Config
from vehicle_idm import IDMVehicle

class Simulator:

    def __init__(self, config: Config):
        self.config = config
        self.vehicles = []


    def run(self):
        """Main simulation loop."""
        number_of_vehicles = 0
        time_generation_last = 0

        dt = self.config.simulation_time_step  # e.g., 0.1 seconds
        num_steps = int((self.config.time_max - 1) / dt) + 1

        for i in range(num_steps):
            t = 1 + i * dt  # simulation time starts at t = 1

            # Print every 100 seconds
            if abs(t % 100) < 1e-6:
                print(f"step: {i}, time: {int(t)}")

            # 1. Vehicle generation / inflow process
            number_of_vehicles, time_generation_last, self.vehicles = (
                self._generate_vehicles(number_of_vehicles, t, time_generation_last, self.vehicles)
            )

            # 2. Apply road/bottleneck speed limits
            self._check_road(t)

            # 3. Car-following model updates
            self._update_all_acceleration()
            self._update_all_speed()
            self._update_all_position()

            # 4. Record per-vehicle state at this timestep
            self._record_all_state(t)

        # Print summary after simulation completes
        print("\nVehicle Number: ", len(self.vehicles))
        print("Inflow Rate: ", int(len(self.vehicles) / (self.config.time_max - 1) * 3600), " veh/h\n" )


    def _check_road(self, current_time):
        """Call each vehicle's bottleneck/speed-limit checker."""
        for vehicle in self.vehicles:
            vehicle.check_road(current_time)


    def _update_all_acceleration(self):
        """Update acceleration of all vehicles using their car-following rules."""
        for vehicle in self.vehicles:
            vehicle.update_acceleration()


    def _update_all_speed(self):
        """Update speeds of all vehicles."""
        for vehicle in self.vehicles:
            vehicle.update_speed()


    def _update_all_position(self):
        """Update positions of all vehicles."""
        for vehicle in self.vehicles:
            vehicle.update_position()


    def _record_all_state(self, t):
        """Record each vehicle’s state at time t."""
        for vehicle in self.vehicles:
            vehicle.record_state(t)


    # ChatGPT: Explain the mechanism for me
    def _generate_vehicles(self, number_of_vehicles, t_current, time_generation_last, vehicles):
        """
        Stochastic vehicle generation process.
        - Vehicles enter the road according to a minimum interval plus
          an exponential random component (extra_interval).
        - Each new vehicle follows the last generated one (v_front).
        """
        # Determine the front vehicle (last in list)
        v_front = vehicles[-1] if vehicles else None

        t_min = self.config.vehicle_min_interval
        extra_interval = self.config.vehicle_extra_interval

        # Initialize the time for the next vehicle arrival
        if not hasattr(self, 'next_generation_time') or self.next_generation_time is None:
            if extra_interval > 0:
                self.next_generation_time = (
                    t_current + t_min + random.expovariate(1.0 / extra_interval)
                )
            else:
                self.next_generation_time = t_current + t_min

        last_generation_time = time_generation_last

        # Generate vehicles as long as time has reached the scheduled generation time
        while self.next_generation_time <= t_current:
            number_of_vehicles += 1

            # Create the new vehicle
            v = IDMVehicle(self.config, number_of_vehicles, v_front)
            vehicles.append(v)
            last_generation_time = self.next_generation_time

            # Update front vehicle pointer
            v_front = vehicles[-1]

            # Schedule next vehicle arrival
            if extra_interval > 0:
                interval = t_min + random.expovariate(1.0 / extra_interval)
            else:
                interval = t_min

            self.next_generation_time = last_generation_time + interval

        return number_of_vehicles, last_generation_time, vehicles
