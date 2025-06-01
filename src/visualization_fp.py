import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter

def plot_results(x_val, data_series, x_type='power', loc='upper left', plot_type='Data Rate', smooth=False, window_length=5, polyorder=2):
    """
    Plot results with options for smoothing and different plot types.

    Parameters:
    - x_val (array): The x-axis values.
    - data_series (list): A list of dictionaries, each containing data and metadata for one series.
    - x_type (str): Type of x-axis ('power' or 'ris_elements').
    - loc (str): Location of the legend.
    - plot_type (str): Type of data being plotted ('Data Rate', 'Energy Efficiency', 'Bar', 'Scatter').
    - smooth (bool): Boolean indicating whether to smooth the curves.
    - window_length (int): Length of the filter window for smoothing.
    - polyorder (int): Order of the polynomial used for smoothing.
    """
    plt.figure(figsize=(10, 6))
    
    # Set x-axis label
    if x_type == 'power':
        plt.xlabel("Maximum available Transmit Power (dBm)", fontsize=15, fontweight='bold')
    else:
        plt.xlabel("Number of RIS elements", fontsize=15, fontweight='bold')
    
    # Set y-axis label
    if plot_type in ['Data Rate', 'Energy Efficiency']:
        ylabel = "Data Rate (bps/Hz)" if plot_type == 'Data Rate' else "Energy Efficiency (bits/J)"
        plt.ylabel(ylabel, fontsize=15, fontweight='bold')
    
    for series in data_series:
        # Apply smoothing if enabled
        if smooth:
            if window_length % 2 == 0:
                window_length += 1
            smoothed_data = savgol_filter(series['data'], window_length, polyorder)
            y_data = smoothed_data
        else:
            y_data = series['data']
        
        # Plot according to the plot type
        if plot_type in ['Data Rate', 'Energy Efficiency']:
            plt.plot(x_val, y_data, label=series['label'], color=series['color'], marker=series['marker'], markersize=8, linewidth=series.get('line_width', 3))
        elif plot_type == 'Bar':
            plt.bar(x_val, y_data, label=series['label'], color=series['color'])
        elif plot_type == 'Scatter':
            plt.scatter(x_val, y_data, label=series['label'], color=series['color'], marker=series['marker'], s=50)
    
    plt.tick_params(axis='both', labelsize=15)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    legend = plt.legend(fontsize=15, loc=loc)
    legend.get_frame().set_facecolor('white')
    plt.show()

def plot_3d_positions(RIS, Tx, Rx_B, Rx_E):
    """
    Plot 3D positions of RIS, UEs, Bob, and Eve.

    Parameters:
    - RIS (array): Position of the RIS.
    - Tx (array): Positions of the UEs.
    - Rx_B (array): Position of Bob.
    - Rx_E (array): Position of Eve.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot RIS
    ax.scatter(RIS[0], RIS[1], RIS[2], c='purple', marker='s', s=200, label='RIS')
    ax.text(RIS[0], RIS[1], RIS[2] + 0.5, 'RIS', color='purple', fontsize=10)
    
    # Plot UEs
    for i, ue in enumerate(Tx):
        ax.scatter(ue[0], ue[1], ue[2], c='blue', marker='o', s=200, label=f'UE {i+1}')
        ax.text(ue[0], ue[1], ue[2] + 0.5, f'UE {i+1}', color='blue', fontsize=10)
    
    # Plot Bob
    ax.scatter(Rx_B[0], Rx_B[1], Rx_B[2], c='green', marker='^', s=200, label='Bob')
    ax.text(Rx_B[0], Rx_B[1], Rx_B[2] + 0.5, 'Bob', color='green', fontsize=10)
    
    # Plot Eve
    ax.scatter(Rx_E[0], Rx_E[1], Rx_E[2], c='red', marker='P', s=200, label='Eve')
    ax.text(Rx_E[0], Rx_E[1], Rx_E[2] + 0.5, 'Eve', color='red', fontsize=10)

    ax.grid(True)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('3D Visualization of UEs, RIS, Bob, and Eve', fontsize=14)
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    

# Example usage:
# RIS = np.array([1, 1, 3])  # RIS position (x, y, z)
# Tx = np.array([[2, 2, 1], [3, 3, 1.5], [4, 1, 2]])  # UE positions (x, y, z)
# Rx_B = np.array([2, 3, 0])  # Bob position (x, y, z)
# Rx_E = np.array([3, 1, 0])  # Eve position (x, y, z)
# plot_3d_positions(RIS, Tx, Rx_B, Rx_E)

# Example usage:
# Ptmax_dbm = np.linspace(-30, 0, 100)  # Example x-axis values
# data_series = [
#     {'data': np.random.rand(100) * 10, 'label': 'Curve 1', 'color': 'red', 'marker': '>'},
#     {'data': np.random.rand(100) * 15, 'label': 'Curve 2', 'color': 'blue', 'marker': 'o'}
# ]
# plot_results(Ptmax_dbm, data_series, smooth=True)

# Data Setup
# Ptmax_dbm = np.linspace(-30, 0, 10)  # Example power range

# Data Rate or Energy Efficiency series examples
# data_rate_series = [
#     {'data': np.random.rand(10), 'label': '(a) SR Max. - Bob', 'color': 'red', 'marker': '>'},
#     {'data': np.random.rand(10), 'label': '(b) SR with GEE Max. - Bob', 'color': 'blue', 'marker': 'o'},
# ]

# energy_efficiency_series = [
#     {'data': np.random.rand(10), 'label': 'GEE Max. - Bob', 'color': 'red', 'marker': '>'},
#     {'data': np.random.rand(10), 'label': 'SEE Max.', 'color': 'magenta', 'marker': '>'},
# ]

# Plotting
# plot_results(Ptmax_dbm, data_rate_series, plot_type='Data Rate')
# plot_results(Ptmax_dbm, energy_efficiency_series, plot_type='Energy Efficiency')
# plot_results(Ptmax_dbm, data_rate_series, plot_type='Bar')
# plot_results(Ptmax_dbm, energy_efficiency_series, plot_type='Scatter')