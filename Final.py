import pandas as pd
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from collections import deque

# Constants
NUM_SIGNALS = 5  # Number of traffic signals in the simulation
TIME_STEPS = 300  # Number of time steps for the simulation (e.g., 300 seconds)
VEHICLE_INFLOW_RATE = 1  # Constant vehicle inflow rate (vehicles per second)
CONGESTION_THRESHOLD = 0.8  # Congestion threshold for the road (80% of the road's capacity)
BASE_GREEN_TIME = 20  # Base green signal time in seconds
MIN_GREEN_TIME = 5  # Minimum green time to avoid infinite reduction
MAX_GREEN_TIME = 60  # Maximum green time to avoid infinite increase

# 1. Generate Dummy Traffic Data
def generate_dummy_traffic_data(num_signals, time_steps, inflow_rate):
    """
    Generates dummy traffic data for a given number of signals and time steps.
    Each signal has a vehicle count and their respective lengths.
    """
    traffic_data = []
    for step in range(time_steps):
        for i in range(num_signals - 1):
            vehicle_count = inflow_rate * (step + 1)  # Constant inflow rate
            vehicle_lengths = [random.randint(3, 12) for _ in range(int(vehicle_count))]
            traffic_data.append({
                'time_step': step,
                'from': f'Signal{i+1}',
                'to': f'Signal{i+2}',
                'vehicles': vehicle_count,
                'vehicle_lengths': vehicle_lengths
            })
    return pd.DataFrame(traffic_data)

# 2. Generate Road Network
def generate_road_network(num_signals=5):
    """Generates a directed road network graph with traffic signals and associated road properties."""
    G = nx.DiGraph()  # Directed graph as traffic is directional
    signals = [f"Signal{i}" for i in range(1, num_signals + 1)]
    G.add_nodes_from(signals)

    # Create edges between consecutive signals with random road attributes
    for i in range(num_signals - 1):
        G.add_edge(signals[i], signals[i+1], 
                   distance=random.randint(100, 300),  # Distance between signals (m)
                   width=random.randint(8, 15),  # Road width (m)
                   capacity_per_meter=3,  # Vehicle capacity per meter of road width
                   queue_capacity=100,  # Queue capacity of each road
                   vehicles=deque())  # Queue of vehicles on the road
    
    return G

# 3. Simulate Traffic Flow and Calculate Congestion
def simulate_traffic_flow(graph, traffic_df):
    """
    Simulates traffic flow, calculates density, and updates traffic data for the road network.
    """
    for index, row in traffic_df.iterrows():
        edge = (row['from'], row['to'])
        vehicles = row['vehicles']
        vehicle_lengths = row['vehicle_lengths']
        graph.edges[edge]['vehicles'].extend(vehicle_lengths)
    
    # Calculate road density based on vehicle lengths, road dimensions, and capacity
    for edge in graph.edges():
        edge_data = graph.edges[edge]
        total_vehicle_length = sum(edge_data['vehicles'])
        road_width = edge_data['width']
        road_length = edge_data['distance']
        capacity = road_width * road_length * edge_data['capacity_per_meter']  # Road capacity
        current_density = total_vehicle_length / capacity
        edge_data['density'] = current_density
        print(f"Road: {edge}, Density: {current_density:.2f}")
    
    return graph

# 4. Calculate Congestion
def calculate_congestion(density, congestion_threshold=0.8):
    """
    Determine the congestion level based on the density of vehicles on a given road.
    """
    if density > congestion_threshold:
        return "high_traffic"
    elif density > congestion_threshold / 2:
        return "medium_traffic"
    else:
        return "low_traffic"

# 5. Signal Control Logic
def adjust_signal_time_based_on_congestion(signal_id, congestion_level, current_green_time):
    """
    Adjust the green signal time based on the congestion level (high, medium, low).
    Limits the green time within a reasonable range (5 to 60 seconds).
    """
    if congestion_level == 'high_traffic':
        green_time = current_green_time * 1.3  # Increase green time by 30% for high congestion
    elif congestion_level == 'medium_traffic':
        green_time = current_green_time * 1.1  # Slight increase for medium congestion
    else:
        green_time = current_green_time * 0.9  # Decrease green time for low congestion
    
    # Enforce limits on the green time
    green_time = max(MIN_GREEN_TIME, min(green_time, MAX_GREEN_TIME))
    
    print(f"Signal {signal_id}: Adjusted green signal time to {green_time:.2f} seconds.")
    return green_time

def adjust_previous_signals_greens(prev_signal_green_time, congestion_level):
    """
    Adjust previous signal's green time based on the congestion of the following signal.
    """
    if congestion_level == 'high_traffic':
        prev_signal_green_time = max(MIN_GREEN_TIME, prev_signal_green_time - 5)  # Reduce by 5 seconds for previous signals
    print(f"Adjusted previous signal green time to: {prev_signal_green_time} seconds.")
    return prev_signal_green_time

# 6. Traffic Management Simulation
def traffic_management_simulation(graph, traffic_df):
    """
    Simulate the traffic signal sequence and adjust green signal times dynamically based on congestion.
    """
    signal_times = {signal: BASE_GREEN_TIME for signal in graph.nodes()}  # Initialize with base green time
    
    # Adjust green signal times based on congestion
    for i, signal in enumerate(list(graph.nodes())[0:-1]):
        edge = (signal, list(graph.nodes())[i+1])
        congestion_level = calculate_congestion(graph.edges[edge]['density'])
        
        # Adjust the green signal time based on congestion level
        green_time = adjust_signal_time_based_on_congestion(signal, congestion_level, signal_times[signal])
        signal_times[signal] = green_time

        # Adjust the previous signal's green time if congestion is detected in the current signal
        if i > 0:
            prev_signal = list(signal_times.keys())[i-1]
            signal_times[prev_signal] = adjust_previous_signals_greens(signal_times[prev_signal], congestion_level)
    
    print("\nFinal Signal Green Times:", signal_times)
    return signal_times, graph

# 7. Traffic Signal Simulation
def traffic_signal_simulation(graph, traffic_df, signal_times):
    """
    Simulate the traffic flow and adjust traffic signal timings for each time step.
    """
    metrics = []
    
    for i, time_step in enumerate(traffic_df['time_step'].unique()):
        df_slice = traffic_df[traffic_df['time_step'] == time_step]
        for index, row in df_slice.iterrows():
            edge = (row['from'], row['to'])
            vehicle_lengths = row['vehicle_lengths']
            graph.edges[edge]['vehicles'].extend(vehicle_lengths)
        
        # Adjust the signal timings based on congestion
        for signal in list(graph.nodes())[0:-1]:
            edge = (signal, list(graph.nodes())[list(graph.nodes()).index(signal)+1])
            congestion_level = calculate_congestion(graph.edges[edge]['density'])
            green_time = adjust_signal_time_based_on_congestion(signal, congestion_level, signal_times[signal])
            # Update the queue capacity based on green signal time
            graph.edges[edge]['queue_capacity'] = (green_time / 60) * graph.edges[edge]['capacity_per_meter'] * graph.edges[edge]['width'] * graph.edges[edge]['distance']
        
        # Record the metrics for this time step
        metrics.append({'time': time_step, 'signal_times': signal_times.copy()})
    
    metrics_df = pd.DataFrame(metrics)
    return metrics_df

# 8. Main Execution
if __name__ == "__main__":
    # Generate dummy traffic data with a constant inflow rate and number of signals
    traffic_df = generate_dummy_traffic_data(NUM_SIGNALS, TIME_STEPS, VEHICLE_INFLOW_RATE)
    
    # Generate the road network with specified number of traffic signals
    graph = generate_road_network(NUM_SIGNALS)
    
    # Simulate traffic flow and calculate congestion on the roads
    graph = simulate_traffic_flow(graph, traffic_df)
    
    # Manage the traffic signals and adjust green signal times based on congestion
    signal_times, graph = traffic_management_simulation(graph, traffic_df)
    
    # Simulate the traffic signal flow with adjusted timings and collect metrics
    metrics_df = traffic_signal_simulation(graph, traffic_df, signal_times)

    # Visualize the signal timings over time using a line plot
    signal_times_df = metrics_df['signal_times'].apply(pd.Series)
    signal_times_df.plot(figsize=(10,6), title="Signal Timings Over Time", marker='o')
    plt.xlabel('Time Steps')
    plt.ylabel('Green Signal Time (seconds)')
    plt.show()

    # Output the final metrics for further analysis
    print(metrics_df)
