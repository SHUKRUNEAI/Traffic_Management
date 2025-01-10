import pandas as pd
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from collections import deque
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load Real Traffic Data
def load_traffic_data(file_path):
    """Loads traffic data from a CSV file."""
    df = pd.read_csv(file_path)
    return df

# 2. Generate Road Network
def generate_road_network(num_signals=5):
    """Generates a road network graph."""
    G = nx.DiGraph()  # Directed graph since traffic is directional
    signals = [f"Signal{i}" for i in range(1, num_signals + 1)]
    G.add_nodes_from(signals)

    for i in range(num_signals - 1):
        G.add_edge(signals[i], signals[i+1], distance=random.randint(100, 300), width=random.randint(8, 15), capacity_per_meter=3, queue_capacity=100, vehicles=deque())
    
    return G

# 3. Simulate Traffic Flow and Calculate Congestion
def simulate_traffic_flow(graph, traffic_df):
    """Simulates traffic flow, generating traffic density data."""
    for index, row in traffic_df.iterrows():
        edge = (row['from'], row['to'])
        vehicles = row['vehicles']
        vehicle_lengths = row['vehicle_lengths']
        graph.edges[edge]['vehicles'].extend(vehicle_lengths)
    
    # Calculate occupancy for all edges
    for edge in graph.edges():
        edge_data = graph.edges[edge]
        total_vehicle_length = sum(edge_data['vehicles'])
        road_width = edge_data['width']
        road_length = edge_data['distance']
        capacity = (road_width * road_length * edge_data['capacity_per_meter'])  # use road width to measure capacity
        current_density = total_vehicle_length / capacity
        edge_data['density'] = current_density
        print(f"Road: {edge}, Density: {current_density}")
    
    return graph

def calculate_congestion(density, congestion_threshold=0.8):
    """
    Calculate the congestion level based on the density.
    If density exceeds threshold, it's considered congested.
    """
    if density > congestion_threshold:
        return "high_traffic"
    elif density > congestion_threshold / 2:
        return "medium_traffic"
    else:
        return "low_traffic"

# 4. Signal Control Logic
def adjust_signal_time_based_on_congestion(signal_id, congestion_level):
    """
    Adjust the green signal time based on the congestion level.
    """
    base_time = 20
    if congestion_level == 'high_traffic':
        green_time = base_time * 1.3  # Increase green signal time
    elif congestion_level == 'medium_traffic':
        green_time = base_time * 1.1  # Standard green time
    else:
        green_time = base_time * 0.9  # Decrease green time for low traffic
    
    print(f"Signal {signal_id}: Adjusted green signal time to {green_time} seconds.")
    return green_time

def adjust_previous_signals_greens(prev_signal_green_time, congestion_level):
    """
    Adjust previous signals' green times based on congestion.
    If next signal is congested, reduce previous signal's green time.
    """
    if congestion_level == 'high_traffic':
        prev_signal_green_time = max(5, prev_signal_green_time - 5)  # Reduce by 5 seconds for previous signals
    print(f"Adjusted previous signal green time: {prev_signal_green_time} seconds.")
    return prev_signal_green_time

# 5. Traffic Management Simulation
def traffic_management_simulation(graph, traffic_df):
    """
    Simulate the traffic signal sequence and adjust green signal times dynamically.
    """
    # Initial signal green times
    signal_times = {signal: 20 for signal in graph.nodes()}  # Starting with 15 seconds each
    
    # Simulate the traffic signal sequence
    for i, signal in enumerate(list(graph.nodes())[0:-1]):
        edge = (signal, list(graph.nodes())[i+1])
        congestion_level = calculate_congestion(graph.edges[edge]['density'])
        
        # Adjust green time based on congestion level
        green_time = adjust_signal_time_based_on_congestion(signal, congestion_level)
        signal_times[signal] = green_time

        # Adjust the previous signal's green time based on congestion in the current signal
        if i > 0:
            prev_signal = list(signal_times.keys())[i-1]
            signal_times[prev_signal] = adjust_previous_signals_greens(signal_times[prev_signal], congestion_level)
    
    print("\nFinal Signal Green Times:", signal_times)
    return signal_times, graph

# 6. Traffic Signal Simulation
def traffic_signal_simulation(graph, traffic_df, signal_times):
    """
    Simulate traffic flow with adaptive signal times, and collect traffic metrics.
    """
    metrics = []
    
    for i, time_step in enumerate(traffic_df['time_step'].unique()):
        df_slice = traffic_df[traffic_df['time_step']==time_step]
        for index, row in df_slice.iterrows():
            edge = (row['from'], row['to'])
            vehicle_lengths = row['vehicle_lengths']
            graph.edges[edge]['vehicles'].extend(vehicle_lengths)
        
        # Adjust the signal timings
        for signal in list(graph.nodes())[0:-1]:
            edge = (signal, list(graph.nodes())[list(graph.nodes()).index(signal)+1])
            congestion_level = calculate_congestion(graph.edges[edge]['density'])
            green_time = adjust_signal_time_based_on_congestion(signal, congestion_level)
            # adjust the queues based on the signal times
            graph.edges[edge]['queue_capacity'] = (signal_times[signal] / 60) * graph.edges[edge]['capacity_per_meter'] * graph.edges[edge]['width'] * graph.edges[edge]['distance']
        
        # Record metrics
        metrics.append({'time': time_step, 'signal_times': signal_times})
    
    metrics_df = pd.DataFrame(metrics)
    return metrics_df

# 7. Main Execution
if __name__ == "__main__":
    # Replace 'your_traffic_data.csv' with your actual traffic data file path
    traffic_data_file = 'your_traffic_data.csv'
    
    traffic_df = load_traffic_data(traffic_data_file)
    num_signals = traffic_df['from'].nunique()
    
    graph = generate_road_network(num_signals)
    graph = simulate_traffic_flow(graph, traffic_df)
    signal_times, graph = traffic_management_simulation(graph, traffic_df)
    metrics_df = traffic_signal_simulation(graph, traffic_df, signal_times)

    # Visualize results
    metrics_df['signal_times'].apply(pd.Series).plot(figsize=(10,6), title="Signal Timings Over Time", marker='o')
    plt.xlabel('Time Steps')
    plt.ylabel('Green Signal Time')
    plt.show()
    print(metrics_df)
