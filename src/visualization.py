import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from config import SystemConfig

class Plotter:
    def __init__(self, x_val, data_series, x_type='power', loc='upper left', plot_type='Data Rate', 
                 smooth=False, window_length=5, polyorder=2):
        self.x_val = x_val
        self.data_series = data_series
        self.x_type = x_type
        self.loc = loc
        self.plot_type = plot_type
        self.smooth = smooth
        self.window_length = window_length
        self.polyorder = polyorder
    
    def plot_results(self):
        plt.figure(figsize=(10, 6))
        self._set_labels()
        
        for series in self.data_series:
            y_data = self._apply_smoothing(series['data']) if self.smooth else series['data']
            self._plot_data(series, y_data)
        
        self._finalize_plot()
        plt.show()

    def _set_labels(self):
        plt.xlabel(self._get_xlabel(), fontsize=15, fontweight='bold')
        plt.ylabel(self._get_ylabel(), fontsize=15, fontweight='bold')

    def _get_xlabel(self):
        return "Maximum available Transmit Power (dBm)" if self.x_type == 'power' else "Number of RIS elements"

    def _get_ylabel(self):
        if self.plot_type == 'Data Rate':
            return "Data Rate (bps/Hz)"
        elif self.plot_type == 'Energy Efficiency':
            return "Energy Efficiency (bits/J)"
        return ""

    def _apply_smoothing(self, data):
        if self.window_length % 2 == 0:
            self.window_length += 1
        return savgol_filter(data, self.window_length, self.polyorder)

    def _plot_data(self, series, y_data):
        plot_methods = {
            'Data Rate': plt.plot,
            'Energy Efficiency': plt.plot,
            'Bar': plt.bar,
            'Scatter': plt.scatter
        }
        plot_method = plot_methods.get(self.plot_type, plt.plot)
        
        if self.plot_type in ['Bar', 'Scatter']:
            plot_method(self.x_val, y_data, label=series['label'], color=series['color'])
        else:
            plot_method(self.x_val, y_data, label=series['label'], color=series['color'], 
                        marker=series['marker'], markersize=8, linewidth=series.get('line_width', 3))

    def _finalize_plot(self):
        plt.tick_params(axis='both', labelsize=15)
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        legend = plt.legend(fontsize=15, loc=self.loc)
        legend.get_frame().set_facecolor('white')


class Plot3DPositions:
    def __init__(self, RIS, Tx, Rx_B, Rx_E):
        self.RIS = RIS
        self.Tx = Tx
        self.Rx_B = Rx_B
        self.Rx_E = Rx_E

    def plot(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        self._plot_point(ax, self.RIS, 'RIS', 'purple', 's')
        for i, ue in enumerate(self.Tx):
            self._plot_point(ax, ue, f'UE {i+1}', 'blue', 'o')
        self._plot_point(ax, self.Rx_B, 'Bob', 'green', '^')
        self._plot_point(ax, self.Rx_E, 'Eve', 'red', 'P')

        self._finalize_plot(ax)
        plt.show()

    def _plot_point(self, ax, point, label, color, marker):
        ax.scatter(point[0], point[1], point[2], c=color, marker=marker, s=200, label=label)
        ax.text(point[0], point[1], point[2] + 0.5, label, color=color, fontsize=10)

    def _finalize_plot(self, ax):
        ax.grid(True)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title('3D Visualization of UEs, RIS, Bob, and Eve', fontsize=14)
        ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')


# Example Usage
if __name__ == "__main__":
    
    # Initialize system configuration
    config = SystemConfig()
    
    # Load the results
    results = np.load(config.OUTPUT_FILE, allow_pickle=True)['arr_0'].item()

    # Extract data for plotting
    x_val = results['x_val']
    data_series = results['data_series']

    # Plot results
    Plotter.plot_results(x_val, data_series, x_type='power', plot_type='Data Rate', smooth=True)
    Plotter.plot_results(x_val, data_series, x_type='ris_elements', plot_type='Energy Efficiency', smooth=True)

    # # Example data for testing
    # x_val = np.linspace(0, 10, 100)
    # data_series = [
    #     {'data': np.sin(x_val), 'label': 'Sine Wave', 'color': 'blue', 'marker': 'o'},
    #     {'data': np.cos(x_val), 'label': 'Cosine Wave', 'color': 'red', 'marker': 'x'}
    # ]
    # plotter = Plotter(x_val, data_series, x_type='power', plot_type='Data Rate', smooth=True)
    # plotter.plot_results()

    # # Example data for 3D plotting
    # RIS = [0, 0, 10]
    # Tx = [[10, 0, 0], [0, 10, 0], [-10, 0, 0], [0, -10, 0]]
    # Rx_B = [5, 5, 0]
    # Rx_E = [-5, -5, 0]
    # plot3d = Plot3DPositions(RIS, Tx, Rx_B, Rx_E)
    # plot3d.plot()
