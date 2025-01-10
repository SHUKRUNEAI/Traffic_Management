import numpy as np
import random
import matplotlib.pyplot as plt

class TrafficSignal:
    def __init__(self, signal_id, green_time=30, amber_time = 5):
        self.signal_id = signal_id
        self.green_time = green_time # base green time
        self.amber_time = amber_time # base amber time
        self.current_light = "green"
        self.time_elapsed = 0 # keeps track of how long the current light has been going
        self.previous_vehicles = 0
        self.congestion_factor = 1 # how much traffic is building up near the light
        self.history = [] # keeps record of the state and how long the lights were

    def update(self, vehicle_count, min_green_time = 15, max_green_time = 60, congestion_threshold = 1.2):
        """Adjust green time based on current traffic."""

        self.history.append({"signal": self.signal_id,
                           "light": self.current_light,
                           "time": self.time_elapsed,
                            "vehicle": vehicle_count,
                             "congestion": self.congestion_factor
                            })

        self.time_elapsed += 1

        if self.current_light == "green" and self.time_elapsed > self.green_time:
            self.current_light = "amber"
            self.time_elapsed = 0

        if self.current_light == "amber" and self.time_elapsed > self.amber_time:
            self.current_light = "red"
            self.time_elapsed = 0
            #calculate congestion
            self.congestion_factor = vehicle_count / (self.previous_vehicles + 1e-6)
            self.previous_vehicles = vehicle_count
        if self.current_light == "red":
            self.current_light = "green"
            self.time_elapsed = 0


        if self.current_light == "green":
            if self.congestion_factor > congestion_threshold :
                self.green_time += 5  # Increase green time
                if self.green_time > max_green_time:
                  self.green_time = max_green_time
            elif self.congestion_factor < 1:
                self.green_time -= 3 # reduce time slightly if there isn't much congestion
                if self.green_time < min_green_time:
                  self.green_time = min_green_time

class RoadSection:
    def __init__(self, section_id, length, width, capacity_factor=0.8):
        self.section_id = section_id
        self.length = length # in whatever units
        self.width = width # also in whatever unit
        self.capacity_factor = capacity_factor
        self.vehicles = [] # list of current vehicles on the section
        self.max_capacity = int(self.width * self.length * self.capacity_factor)

    def add_vehicle(self, vehicle):
        if len(self.vehicles) < self.max_capacity:
            self.vehicles.append(vehicle)

    def move_vehicles(self, next_signal):
        """Move vehicles towards the next signal, returns vehicles that pass the signal"""
        passed_vehicles = []
        for vehicle in self.vehicles:
            vehicle.travel_progress += vehicle.speed
            if vehicle.travel_progress >= self.length:
                passed_vehicles.append(vehicle)

        self.vehicles = [v for v in self.vehicles if v not in passed_vehicles]
        return passed_vehicles

class Vehicle:
      def __init__(self, vehicle_type, dimension, speed = 10):
        self.vehicle_type = vehicle_type
        self.dimension = dimension # arbitrary unit
        self.speed = speed
        self.travel_progress = 0

# Set up road section

road = RoadSection("main_road", length = 100, width = 5)

# Set up traffic signals
signal1 = TrafficSignal(signal_id="signal1", green_time = 20)
signal2 = TrafficSignal(signal_id = "signal2", green_time = 30)

# Simulation parameters
simulation_time = 200
time = 0
traffic_flow_per_period = 2 # max vehicles that can come per time step
signal_state_history = []


# Define vehicle types and their properties (for testing purposes)
vehicle_types = {
    "office_worker": {"dimension": 1, "speed": 10},
    "student": {"dimension": 0.8, "speed": 8},
    "bus": {"dimension": 2, "speed": 5},
}

# Simulation Loop
while time < simulation_time:
    # 1. Create new vehicles (randomly for now)
    num_new_vehicles = random.randint(0, traffic_flow_per_period)
    for _ in range(num_new_vehicles):
        type = random.choice(list(vehicle_types.keys()))
        vehicle = Vehicle(vehicle_type = type, dimension = vehicle_types[type]["dimension"], speed = vehicle_types[type]["speed"])
        road.add_vehicle(vehicle)

    # 2. Move vehicles toward next signal and record which ones left the road section
    vehicles_passing_signal1 = road.move_vehicles(signal1)


    #3. Update signals
    signal1.update(vehicle_count = len(vehicles_passing_signal1))

    #4. Record what signal state was for all traffic signals
    signal_state_history.append({"time": time,
                                "signal1": {"green": (signal1.current_light == "green"), "amber": (signal1.current_light == "amber"), "red": (signal1.current_light == "red"), "greentime": signal1.green_time, "congestion": signal1.congestion_factor, "vehicle": len(vehicles_passing_signal1)}
                                # you can add more signals like this.
                              })

    time+=1

# Visualization

signal_data = {"time": [], "green_time_1": [], "congestion_1": []}

for record in signal_state_history:
    signal_data["time"].append(record["time"])
    signal_data["green_time_1"].append(record["signal1"]["greentime"])
    signal_data["congestion_1"].append(record["signal1"]["congestion"])

plt.figure(figsize=(10, 6))
plt.plot(signal_data["time"], signal_data["green_time_1"], label = "Signal 1 Green Time")
plt.xlabel('Time Step')
plt.ylabel('Green Time')
plt.title('Adaptive Signal Timing Simulation')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(signal_data["time"], signal_data["congestion_1"], label = "Signal 1 Congestion Factor")
plt.xlabel('Time Step')
plt.ylabel('Congestion Factor')
plt.title('Adaptive Signal Timing Simulation')
plt.legend()
plt.grid(True)
plt.show()

print(signal1.history)
