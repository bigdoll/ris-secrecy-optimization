import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from config import SystemConfig

class Plotter:
    def __init__(self, x_val, data_series, loc='upper left', window_length=5, polyorder=1, x_type='power', plot_type='Data Rate', 
                 smooth=False, combined_plot=False):
        self.x_val = x_val
        self.data_series = data_series
        self.x_type = x_type
        self.loc = loc
        self.plot_type = plot_type
        self.smooth = smooth
        self.window_length = window_length
        self.polyorder = polyorder
        self.combined_plot = combined_plot
    
    def plot_results(self):
        if self.combined_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
            self._set_labels(ax1, ax2)

            for series in self.data_series:
                y_data = self._apply_smoothing(series['data']) if self.smooth else series['data']
                self._plot_data(ax1 if 'pCSI' in series['label'] else ax2, series, y_data)
            
            self._finalize_plot(ax1, ax2)
        else:
            plt.figure(figsize=(10, 6))
            self._set_labels()

            for series in self.data_series:
                y_data = self._apply_smoothing(series['data']) if self.smooth else series['data']
                self._plot_data(plt, series, y_data)
            
            self._finalize_plot()
        plt.show()

    def _set_labels(self, ax1=None, ax2=None):
        xlabel = self._get_xlabel()
        ylabel = self._get_ylabel()

        if self.combined_plot:
            ax1.set_xlabel(xlabel, fontsize=15, fontweight='bold')
            ax1.set_ylabel(ylabel + " (pCSI)", fontsize=15, fontweight='bold')
            ax2.set_xlabel(xlabel, fontsize=15, fontweight='bold')
            ax2.set_ylabel(ylabel + " (sCSI)", fontsize=15, fontweight='bold')
        else:
            plt.xlabel(xlabel, fontsize=15, fontweight='bold')
            plt.ylabel(ylabel, fontsize=15, fontweight='bold')

    def _get_xlabel(self):
        if self.x_type == 'power':
            return "Maximum available Transmit Power - Ptmax (dBm)"
        elif self.x_type == 'ris':
            return "Number of RIS elements (N)"
        elif self.x_type == 'PRmax':
            return "Maximum RF Power of the RIS - PRmax (dBm)"
        elif self.x_type == 'a':
            return "Maximum Gain of the RIS -  a (dB)"
        elif self.x_type == 'Pcn':
            return "Static Power Consumption per RIS element - Pcn (dBm)"
        return "Parameter"

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

    def _plot_data(self, ax, series, y_data):
        plot_methods = {
            'Data Rate': ax.plot,
            'Energy Efficiency': ax.plot,
            'Bar': ax.bar,
            'Scatter': ax.scatter
        }
        plot_method = plot_methods.get(self.plot_type, ax.plot)
        
        if self.plot_type in ['Bar', 'Scatter']:
            plot_method(self.x_val, y_data, label=series['label'], color=series['color'])
        else:
            plot_method(self.x_val, y_data, label=series['label'], color=series['color'], 
                        marker=series['marker'], markersize=5, linewidth=series.get('line_width', 3))

    def _finalize_plot(self, ax1=None, ax2=None):
        if self.combined_plot:
            ax1.tick_params(axis='both', labelsize=15)
            ax2.tick_params(axis='both', labelsize=15)
            ax1.legend(fontsize=15, loc=self.loc)
            ax2.legend(fontsize=15, loc=self.loc)
        else:
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
        plt.show(block=True)

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


def get_plot_config():
    use_default = input("Do you want to use the default plot configuration? (yes/no): ").strip().lower()
    
    if use_default == 'no':
        loc = input("Enter legend location (default: 'upper left'): ").strip() or 'upper left'
        plot_type = input("Enter plot type ('Data Rate' or 'Energy Efficiency', default: 'Data Rate'): ").strip() or 'Data Rate'
        smooth = input("Do you want to apply smoothing? (yes/no, default: no): ").strip().lower() == 'yes'
        window_length = int(input("Enter window length for smoothing (default: 5): ").strip() or 5)
        polyorder = int(input("Enter polynomial order for smoothing (default: 1): ").strip() or 1)
        bits_range = eval(input("Enter bits range as a list of tuples, e.g., [(1,1), (2,2), (3,3), (4,4)]: "))
        quantized_plot = input("Do you want to plot the Quantization Results as well? (yes/no): ").strip().lower() == 'yes'
        combined_plot = input("Do you want combined plot with separate axes? (yes/no): ").strip().lower() == 'yes'
    else:
        loc = 'upper left'
        plot_type = 'Data Rate'
        smooth = True
        window_length = 5
        polyorder = 1
        quantized_plot = True
        bits_range = config.bits_range
        combined_plot_wq = True
        combined_plot_woq = False

    return loc, plot_type, smooth, window_length, polyorder, bits_range, quantized_plot, combined_plot_wq, combined_plot_woq

def load_results(file_path):
    return np.load(file_path, allow_pickle=True)['arr_0'].item()

def plot_data(results_ssr, results_see, x_val, keys_to_average, loc, window_length, polyorder, plot_type, x_type, combined_plot_wq, combined_plot_woq, bits_range, quantized_plot=True, smooth=False):
    results_avg_ssr = Utils.average_results(results_ssr, keys_to_average)
    results_avg_see = Utils.average_results(results_see, keys_to_average)

    if quantized_plot:
        colors = ['cyan', 'green', 'blue', 'magenta']
        markers = ['x', '*', '^', 's']

        rate_series_Q_pcsi = [
            {'data': results_avg_ssr['ssr_sol_pcsi'], 'label': 'SSR Max. - pCSI', 'color': 'red', 'marker': 'o'}
        ] + [{'data': results_avg_ssr['ssr_sol_Q_pcsi'][bits], 'label': f'SSR_Q Max. - pCSI - {bits} bits', 'color': colors[i % len(colors)], 'marker': markers[i % len(markers)]} for i, bits in enumerate(bits_range)]

        rate_series_Q_scsi = [
            {'data': results_avg_ssr['ssr_sol_scsi'], 'label': 'SSR Max. - sCSI', 'color': 'red','marker': 'o'}
        ] + [{'data': results_avg_ssr['ssr_sol_Q_scsi'][bits], 'label': f'SSR_Q Max. - sCSI - {bits} bits', 'color': colors[i % len(colors)], 'marker': markers[i % len(markers)]} for i, bits in enumerate(bits_range)]

        energy_series_Q_pcsi = [
            {'data': results_avg_see['see_sol_pcsi'], 'label': 'SEE Max. - pCSI', 'color': 'red', 'marker': 'o'}
        ] + [{'data': results_avg_see['see_sol_Q_pcsi'][bits], 'label': f'SEE_Q Max. - sCSI - {bits} bits', 'color': colors[i % len(colors)], 'marker': markers[i % len(markers)]} for i, bits in enumerate(bits_range)]

        energy_series_Q_scsi = [
            {'data': results_avg_see['see_sol_scsi'], 'label': 'SEE Max. - sCSI', 'color': 'red', 'marker': 'o'}
        ] + [{'data': results_avg_see['see_sol_Q_scsi'][bits], 'label': f'SEE_Q Max. - sCSI - {bits} bits', 'color': colors[i % len(colors)], 'marker': markers[i % len(markers)]} for i, bits in enumerate(bits_range)]

        if combined_plot_wq:
            plotter_rate_Q_combined = Plotter(x_val, rate_series_Q_pcsi + rate_series_Q_scsi, loc, window_length, polyorder, x_type=x_type, plot_type='Data Rate', smooth=smooth, combined_plot=True)
            plotter_energy_Q_combined = Plotter(x_val, energy_series_Q_pcsi + energy_series_Q_scsi, loc, window_length, polyorder, x_type=x_type, plot_type='Energy Efficiency', smooth=smooth, combined_plot=True)
            plotter_rate_Q_combined.plot_results()
            plotter_energy_Q_combined.plot_results()
        else:
            plotter_rate_Q_pcsi = Plotter(x_val, rate_series_Q_pcsi, loc, window_length, polyorder, x_type=x_type, plot_type='Data Rate', smooth=smooth)
            plotter_rate_Q_scsi = Plotter(x_val, rate_series_Q_scsi, loc, window_length, polyorder, x_type=x_type, plot_type='Data Rate', smooth=smooth)
            plotter_energy_Q_pcsi = Plotter(x_val, energy_series_Q_pcsi, loc, window_length, polyorder, x_type=x_type, plot_type='Energy Efficiency', smooth=smooth)
            plotter_energy_Q_scsi = Plotter(x_val, energy_series_Q_scsi, loc, window_length, polyorder, x_type=x_type, plot_type='Energy Efficiency', smooth=smooth)

            plotter_rate_Q_pcsi.plot_results()
            plotter_rate_Q_scsi.plot_results()
            plotter_energy_Q_pcsi.plot_results()
            plotter_energy_Q_scsi.plot_results()

    rate_series = [
        {'data': results_avg_ssr['ssr_sol_pcsi'], 'label': 'SSR Max. - pCSI', 'color': 'blue', 'marker': 'o'},
        {'data': results_avg_ssr['ssr_sol_scsi'], 'label': 'SSR Max. - sCSI', 'color': 'red', 'marker': 'x'},
        {'data': results_avg_see['ssr_sol_pcsi'], 'label': 'SSR with SEE Max. - pCSI', 'color': 'green', 'marker': 'o'},
        {'data': results_avg_see['ssr_sol_scsi'], 'label': 'SSR with SEE Max. - sCSI', 'color': 'magenta', 'marker': 'x'},
        {'data': results_avg_ssr['ssr_uniform_pcsi'], 'label': 'SSR with random phases - pCSI', 'color': 'cyan', 'marker': 'o'},
        {'data': results_avg_ssr['ssr_uniform_scsi'], 'label': 'SSR with random phases - sCSI', 'color': 'black', 'marker': 'x'}
    ]

    energy_series = [
        {'data': results_avg_see['see_sol_pcsi'], 'label': 'SEE Max. - pCSI', 'color': 'blue', 'marker': 'o'},
        {'data': results_avg_see['see_sol_scsi'], 'label': 'SEE Max. - sCSI', 'color': 'red', 'marker': 'x'},
        {'data': results_avg_ssr['see_sol_pcsi'], 'label': 'SEE with SSR Max. - pCSI', 'color': 'green', 'marker': 'o'},
        {'data': results_avg_ssr['see_sol_scsi'], 'label': 'SEE with SSR Max. - sCSI', 'color': 'magenta', 'marker': 'x'},
        {'data': results_avg_see['see_uniform_pcsi'], 'label': 'SEE with random phases - pCSI', 'color': 'cyan', 'marker': 'o'},
        {'data': results_avg_see['see_uniform_scsi'], 'label': 'SEE with random phases - sCSI', 'color': 'black', 'marker': 'x'}
    ]

    if combined_plot_woq:
        plotter_rate_combined = Plotter(x_val, rate_series, loc, window_length, polyorder, x_type=x_type, plot_type='Data Rate', smooth=smooth, combined_plot=True)
        plotter_energy_combined = Plotter(x_val, energy_series, loc, window_length, polyorder, x_type=x_type, plot_type='Energy Efficiency', smooth=smooth, combined_plot=True)
        plotter_rate_combined.plot_results()
        plotter_energy_combined.plot_results()
    else:
        plotter_rate = Plotter(x_val, rate_series, loc, window_length, polyorder, x_type=x_type, plot_type='Data Rate', smooth=smooth)
        plotter_energy = Plotter(x_val, energy_series, loc, window_length, polyorder, x_type=x_type, plot_type='Energy Efficiency', smooth=smooth)
        
        plotter_rate.plot_results()
        plotter_energy.plot_results()

def main():
    print("""
    ##############################################################
    #    Resource Allocation and Energy Efficiency Management    #
    #      in RIS-Aided Physical Layer Security Communication     #
    ##############################################################
    """)

    print("""
    Select the type of simulation to plot:
    1. SSR, SEE Vs Ptmax
    2. SSR, SEE Vs N
    3. SSR, SEE Vs PRmax
    4. SSR, SEE Vs a
    5. SSR, SEE Vs Pcn
    6. Customize (Select multiple parameters to plot)
    """)

    plot_option = int(input("Enter the number corresponding to the plot type: "))
    plot_both = input("Do you want to plot both SSR and SEE? (yes/no): ").strip().lower() == 'yes'
    
    if plot_both:
        file_path_ssr = input("Enter the path to the SSR results file: ").strip()
        file_path_see = input("Enter the path to the SEE results file: ").strip()
        results_ssr = load_results(file_path_ssr)
        results_see = load_results(file_path_see)
    else:
        file_path = input("Enter the path to the results file: ").strip()
        results = load_results(file_path)
    
    keys_to_average = {
        "sr_uniform_Bob_pcsi", "sr_uniform_Bob_scsi", "sr_uniform_Eve_pcsi",
        "sr_uniform_Eve_scsi", "ssr_uniform_pcsi", "ssr_uniform_scsi", "gee_uniform_Bob_pcsi", "gee_uniform_Bob_scsi",
        "gee_uniform_Eve_pcsi", "gee_uniform_Eve_scsi", "see_uniform_pcsi", "see_uniform_scsi", "ssr_sol_pcsi", "ssr_sol_Q_pcsi",
        "ssr_sol_scsi", "ssr_sol_Q_scsi", "see_sol_pcsi", "see_sol_Q_pcsi", "see_sol_scsi", "see_sol_Q_scsi", "iteration_altopt_pcsi", "iteration_altopt_scsi",
        "iteration_p_pcsi", "iteration_p_scsi", "iteration_gamma_pcsi", "iteration_gamma_scsi",
        "time_complexity_altopt_pcsi", "time_complexity_altopt_scsi", "time_complexity_p_pcsi",
        "time_complexity_p_scsi", "time_complexity_gamma_pcsi", "time_complexity_gamma_scsi"
    }

    if plot_option in [1, 2, 3, 4, 5]:
        param = {1: ('Ptmax',), 2: ('N',), 3: ('PRmax',), 4: ('a',), 5: ('Pcn',)}[plot_option]
        x_type = {1: 'power', 2: 'ris', 3: 'PRmax', 4: 'a', 5: 'Pcn'}[plot_option]
        if plot_both:
            x_val = results_ssr[0][param]
        else:
            x_val = results[0][param]

        # plot_type = input(f"Do you want to plot {param}? (yes/no): ").strip().lower() == 'yes'
        # combined_plot = input("Do you want combined plot with separate axes? (yes/no): ").strip().lower() == 'yes'

        loc, plot_type, smooth, window_length, polyorder, bits_range, quantized_plot, combined_plot_wq, combined_plot_woq = get_plot_config()

        if plot_both:
            plot_data(results_ssr, results_see, x_val, keys_to_average, loc, window_length, polyorder, plot_type, x_type, combined_plot_wq, combined_plot_woq, bits_range, quantized_plot, smooth)
            # plot_data(results_ssr, results_see, x_val, keys_to_average, loc, window_length, polyorder, plot_type, x_type, combined_plot, bits_range, quantized_plot, smooth)
        else:
            plot_data(results, x_val, keys_to_average, loc, window_length, polyorder, plot_type, x_type, combined_plot_wq, combined_plot_woq, bits_range, quantized_plot, smooth)

    elif plot_option == 6:
        param1 = input("Enter the first parameter to vary (e.g., N): ").strip()
        param2 = input("Enter the second parameter to vary (e.g., Pcn): ").strip()
        x_type = param2  # Use the second parameter as x_type
        if plot_both:
            x_val = results_ssr[0][param2]
        else:
            x_val = results[0][param2]

        loc, plot_type, smooth, window_length, polyorder, bits_range, quantized_plot, combined_plot_wq, combined_plot_woq = get_plot_config()

        for p1_value in (results_ssr if plot_both else results)[param1]:
            sub_results_ssr = {k: v for k, v in results_ssr.items() if k.startswith(f'{param1}_{p1_value}')} if plot_both else None
            sub_results_see = {k: v for k, v in results_see.items() if k.startswith(f'{param1}_{p1_value}')} if plot_both else None
            sub_results = {k: v for k, v in results.items() if k.startswith(f'{param1}_{p1_value}')} if not plot_both else None

            if plot_both:
                plot_data(sub_results_ssr, sub_results_see, x_val, keys_to_average, loc, window_length, polyorder, plot_type, x_type, combined_plot_wq, combined_plot_woq, bits_range, quantized_plot, smooth)
                # plot_data(sub_results_ssr, sub_results_see, x_val, keys_to_average, loc, window_length, polyorder, plot_type, x_type, combined_plot, bits_range, quantized_plot, smooth)
            else:
                plot_data(sub_results, x_val, keys_to_average, loc, window_length, polyorder, plot_type, x_type, combined_plot_wq, combined_plot_woq, bits_range, quantized_plot, smooth)
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    from utils import Utils
    
    config = SystemConfig()
    main()



# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.signal import savgol_filter
# from config import SystemConfig

# class Plotter:
#     def __init__(self, x_val, data_series, loc='upper left', window_length=5,  polyorder=1,  x_type='power', plot_type='Data Rate', 
#                  smooth=False, combined_plot=False):
#         self.x_val = x_val
#         self.data_series = data_series
#         self.x_type = x_type
#         self.loc = loc
#         self.plot_type = plot_type
#         self.smooth = smooth
#         self.window_length = window_length
#         self.polyorder = polyorder
#         self.combined_plot = combined_plot
    
#     def plot_results(self):
#         if self.combined_plot:
#             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
#             self._set_labels(ax1, ax2)

#             for series in self.data_series:
#                 y_data = self._apply_smoothing(series['data']) if self.smooth else series['data']
#                 self._plot_data(ax1 if 'pCSI' in series['label'] else ax2, series, y_data)
            
#             self._finalize_plot(ax1, ax2)
#         else:
#             plt.figure(figsize=(10, 6))
#             self._set_labels()

#             for series in self.data_series:
#                 y_data = self._apply_smoothing(series['data']) if self.smooth else series['data']
#                 self._plot_data(plt, series, y_data)
            
#             self._finalize_plot()
#         plt.show()

#     def _set_labels(self, ax1=None, ax2=None):
#         xlabel = self._get_xlabel()
#         ylabel = self._get_ylabel()

#         if self.combined_plot:
#             ax1.set_xlabel(xlabel, fontsize=15, fontweight='bold')
#             ax1.set_ylabel(ylabel + " (pCSI)", fontsize=15, fontweight='bold')
#             ax2.set_xlabel(xlabel, fontsize=15, fontweight='bold')
#             ax2.set_ylabel(ylabel + " (sCSI)", fontsize=15, fontweight='bold')
#         else:
#             plt.xlabel(xlabel, fontsize=15, fontweight='bold')
#             plt.ylabel(ylabel, fontsize=15, fontweight='bold')

#     def _get_xlabel(self):
#         if self.x_type == 'power':
#             return "Maximum available Transmit Power - Ptmax (dBm)"
#         elif self.x_type == 'ris':
#             return "Number of RIS elements (N)"
#         elif self.x_type == 'PRmax':
#             return "Maximum RF Power of the RIS - PRmax (dBm)"
#         elif self.x_type == 'a':
#             return "Maximum Gain of the RIS -  a (dB)"
#         elif self.x_type == 'Pcn':
#             return "Static Power Consumption per RIS element - Pcn (dBm)"
#         return "Parameter"

#     def _get_ylabel(self):
#         if self.plot_type == 'Data Rate':
#             return "Data Rate (bps/Hz)"
#         elif self.plot_type == 'Energy Efficiency':
#             return "Energy Efficiency (bits/J)"
#         return ""

#     def _apply_smoothing(self, data):
#         if self.window_length % 2 == 0:
#             self.window_length += 1
#         return savgol_filter(data, self.window_length, self.polyorder)

#     def _plot_data(self, ax, series, y_data):
#         plot_methods = {
#             'Data Rate': ax.plot,
#             'Energy Efficiency': ax.plot,
#             'Bar': ax.bar,
#             'Scatter': ax.scatter
#         }
#         plot_method = plot_methods.get(self.plot_type, ax.plot)
        
#         if self.plot_type in ['Bar', 'Scatter']:
#             plot_method(self.x_val, y_data, label=series['label'], color=series['color'])
#         else:
#             plot_method(self.x_val, y_data, label=series['label'], color=series['color'], 
#                         marker=series['marker'], markersize=5, linewidth=series.get('line_width', 3))

#     def _finalize_plot(self, ax1=None, ax2=None):
#         if self.combined_plot:
#             ax1.tick_params(axis='both', labelsize=15)
#             ax2.tick_params(axis='both', labelsize=15)
#             ax1.legend(fontsize=15, loc=self.loc)
#             ax2.legend(fontsize=15, loc=self.loc)
#         else:
#             plt.tick_params(axis='both', labelsize=15)
#             plt.xticks(fontweight='bold')
#             plt.yticks(fontweight='bold')
#             legend = plt.legend(fontsize=15, loc=self.loc)
#             legend.get_frame().set_facecolor('white')


# class Plot3DPositions:
#     def __init__(self, RIS, Tx, Rx_B, Rx_E):
#         self.RIS = RIS
#         self.Tx = Tx
#         self.Rx_B = Rx_B
#         self.Rx_E = Rx_E

#     def plot(self):
#         fig = plt.figure(figsize=(8, 6))
#         ax = fig.add_subplot(111, projection='3d')

#         self._plot_point(ax, self.RIS, 'RIS', 'purple', 's')
#         for i, ue in enumerate(self.Tx):
#             self._plot_point(ax, ue, f'UE {i+1}', 'blue', 'o')
#         self._plot_point(ax, self.Rx_B, 'Bob', 'green', '^')
#         self._plot_point(ax, self.Rx_E, 'Eve', 'red', 'P')

#         self._finalize_plot(ax)
#         plt.show(block=True)

#     def _plot_point(self, ax, point, label, color, marker):
#         ax.scatter(point[0], point[1], point[2], c=color, marker=marker, s=200, label=label)
#         ax.text(point[0], point[1], point[2] + 0.5, label, color=color, fontsize=10)

#     def _finalize_plot(self, ax):
#         ax.grid(True)
#         ax.set_xlabel('X', fontsize=12)
#         ax.set_ylabel('Y', fontsize=12)
#         ax.set_zlabel('Z', fontsize=12)
#         ax.set_title('3D Visualization of UEs, RIS, Bob, and Eve', fontsize=14)
#         ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')


# def get_plot_config():
#     use_default = input("Do you want to use the default plot configuration? (yes/no): ").strip().lower()
    
#     if use_default == 'no':
#         loc = input("Enter legend location (default: 'upper left'): ").strip() or 'upper left'
#         plot_type = input("Enter plot type ('Data Rate' or 'Energy Efficiency', default: 'Data Rate'): ").strip() or 'Data Rate'
#         smooth = input("Do you want to apply smoothing? (yes/no, default: no): ").strip().lower() == 'yes'
#         window_length = int(input("Enter window length for smoothing (default: 5): ").strip() or 5)
#         polyorder = int(input("Enter polynomial order for smoothing (default: 1): ").strip() or 1)
#         bits_range = eval(input("Enter bits range as a list of tuples, e.g., [(1,1), (2,2), (3,3), (4,4)]: "))
#         quantized_plot = input("Do you want to plot the Quantization Results as well? (yes/no): ").strip().lower() == 'yes'
#         combined_plot = input("Do you want combined plot with separate axes? (yes/no): ").strip().lower() == 'yes'
#     else:
#         loc = 'upper left'
#         plot_type = 'Data Rate'
#         smooth = False
#         window_length = 5
#         polyorder = 1
#         quantized_plot =  True
#         bits_range = config.bits_range
#         combined_plot = False
        

#     return loc, plot_type, smooth, window_length, polyorder, bits_range, quantized_plot, combined_plot

# def load_results(file_path):
#     return np.load(file_path, allow_pickle=True)['arr_0'].item()

# def plot_data(results, x_val, keys_to_average, loc, window_length, polyorder, plot_type, x_type, combined_plot, bits_range, quantized_plot = True, smooth =  False):
#     results_avg = Utils.average_results(results, keys_to_average)

#     if quantized_plot:
#         colors = ['cyan', 'green', 'blue', 'magenta']
#         markers = ['x', '*', '^', 's']

#         rate_series_Q_pcsi = [
#             {'data': results_avg['ssr_sol_pcsi'], 'label': 'SSR Max.', 'color': 'red', 'marker': 'o'}
#         ] + [{'data': results_avg['ssr_sol_Q_pcsi'][bits], 'label': f'SSR_Q Max. - {bits} bits', 'color': colors[i % len(colors)], 'marker': markers[i % len(markers)]} for i, bits in enumerate(bits_range)]

#         rate_series_Q_scsi = [
#             {'data': results_avg['ssr_sol_scsi'], 'label': 'SSR Max.', 'color': 'red', 'marker': 'o'}
#         ] + [{'data': results_avg['ssr_sol_Q_scsi'][bits], 'label': f'SSR_Q Max. - {bits} bits', 'color': colors[i % len(colors)], 'marker': markers[i % len(markers)]} for i, bits in enumerate(bits_range)]

#         energy_series_Q_pcsi = [
#             {'data': results_avg['see_sol_pcsi'], 'label': 'SEE Max.', 'color': 'red', 'marker': 'o'}
#         ] + [{'data': results_avg['see_sol_Q_pcsi'][bits], 'label': f'SEE_Q Max. - {bits} bits', 'color'
# : colors[i % len(colors)], 'marker': markers[i % len(markers)]} for i, bits in enumerate(bits_range)]

#         energy_series_Q_scsi = [
#             {'data': results_avg['see_sol_scsi'], 'label': 'SEE Max.', 'color': 'red', 'marker': 'o'}
#         ] + [{'data': results_avg['see_sol_Q_scsi'][bits], 'label': f'SEE_Q Max. - {bits} bits', 'color': colors[i % len(colors)], 'marker': markers[i % len(markers)]} for i, bits in enumerate(bits_range)]

#         if combined_plot:
#             plotter_rate_Q_combined = Plotter(x_val, rate_series_Q_pcsi + rate_series_Q_scsi, loc, window_length, polyorder, x_type=x_type, plot_type=plot_type, smooth=smooth, combined_plot=True)
#             plotter_energy_Q_combined = Plotter(x_val, energy_series_Q_pcsi + energy_series_Q_scsi, loc, window_length, polyorder, x_type=x_type, plot_type=plot_type, smooth=smooth, combined_plot=True)
#             plotter_rate_Q_combined.plot_results()
#             plotter_energy_Q_combined.plot_results()
#         else:
#             plotter_rate_Q_pcsi = Plotter(x_val, rate_series_Q_pcsi, loc, window_length, polyorder, x_type=x_type, plot_type=plot_type, smooth=smooth)
#             plotter_rate_Q_scsi = Plotter(x_val, rate_series_Q_scsi, loc, window_length, polyorder, x_type=x_type, plot_type=plot_type, smooth=smooth)
#             plotter_energy_Q_pcsi = Plotter(x_val, energy_series_Q_pcsi, loc, window_length, polyorder, x_type=x_type, plot_type=plot_type, smooth=smooth)
#             plotter_energy_Q_scsi = Plotter(x_val, energy_series_Q_scsi, loc, window_length, polyorder, x_type=x_type, plot_type=plot_type, smooth=smooth)

#             plotter_rate_Q_pcsi.plot_results()
#             plotter_rate_Q_scsi.plot_results()
#             plotter_energy_Q_pcsi.plot_results()
#             plotter_energy_Q_scsi.plot_results()
#     else:
#         rate_series = [
#             {'data': results_avg['ssr_sol_pcsi'], 'label': 'SSR Max. - pCSI', 'color': 'blue', 'marker': 'o'},
#             {'data': results_avg['ssr_sol_scsi'], 'label': 'SSR Max. - sCSI', 'color': 'red', 'marker': 'x'},
#             {'data': results_avg['ssr_uniform_pcsi'], 'label': 'SSR with random phases - pCSI', 'color': 'green', 'marker': 'o'},
#             {'data': results_avg['ssr_uniform_scsi'], 'label': 'SSR with random phases - sCSI', 'color': 'magenta', 'marker': 'x'}
#         ]

#         energy_series = [
#             {'data': results_avg['see_sol_pcsi'], 'label': 'SEE Max. - pCSI', 'color': 'blue', 'marker': 'o'},
#             {'data': results_avg['see_sol_scsi'], 'label': 'SEE Max. - sCSI', 'color': 'red', 'marker': 'x'},
#             {'data': results_avg['see_uniform_pcsi'], 'label': 'SEE with random phases - pCSI', 'color': 'green', 'marker': 'o'},
#             {'data': results_avg['see_uniform_scsi'], 'label': 'SEE with random phases - sCSI', 'color': 'magenta', 'marker': 'x'}
#         ]

#         if combined_plot:
#             plotter_rate_combined = Plotter(x_val, rate_series, loc, window_length, polyorder, x_type=x_type, plot_type=plot_type, smooth=smooth, combined_plot=True)
#             plotter_energy_combined = Plotter(x_val, energy_series, loc, window_length, polyorder, x_type=x_type, plot_type=plot_type, smooth=smooth, combined_plot=True)
#             plotter_rate_combined.plot_results()
#             plotter_energy_combined.plot_results()
#         else:
#             plotter_rate = Plotter(x_val, rate_series, loc, window_length, polyorder, x_type=x_type, plot_type=plot_type, smooth=smooth)
#             plotter_energy = Plotter(x_val, energy_series, loc, window_length, polyorder, x_type=x_type, plot_type=plot_type, smooth=smooth)
            
#             plotter_rate.plot_results()
#             plotter_energy.plot_results()

# def main():
#     print("""
#     ##############################################################
#     #    Resource Allocation and Energy Efficiency Management    #
#     #      in RIS-Aided Physical Layer Security Communication     #
#     ##############################################################
#     """)

#     print("""
#     Select the type of simulation to plot:
#     1. SSR, SEE Vs Ptmax
#     2. SSR, SEE Vs N
#     3. SSR, SEE Vs PRmax
#     4. SSR, SEE Vs a
#     5. SSR, SEE Vs Pcn
#     6. Customize (Select multiple parameters to plot)
#     """)

#     plot_option = int(input("Enter the number corresponding to the plot type: "))
#     file_path = input("Enter the path to the results file: ").strip()

#     results = load_results(file_path)
    
#     keys_to_average = {
#     "sr_uniform_Bob_pcsi", "sr_uniform_Bob_scsi", "sr_uniform_Eve_pcsi",
#     "sr_uniform_Eve_scsi", "ssr_uniform_pcsi", "ssr_uniform_scsi", "gee_uniform_Bob_pcsi", "gee_uniform_Bob_scsi",
#     "gee_uniform_Eve_pcsi", "gee_uniform_Eve_scsi", "see_uniform_pcsi", "see_uniform_scsi", "ssr_sol_pcsi", "ssr_sol_Q_pcsi",
#     "ssr_sol_scsi", "ssr_sol_Q_scsi", "see_sol_pcsi", "see_sol_Q_pcsi", "see_sol_scsi", "see_sol_Q_scsi", "iteration_altopt_pcsi", "iteration_altopt_scsi",
#     "iteration_p_pcsi", "iteration_p_scsi", "iteration_gamma_pcsi", "iteration_gamma_scsi",
#     "time_complexity_altopt_pcsi", "time_complexity_altopt_scsi", "time_complexity_p_pcsi",
#     "time_complexity_p_scsi", "time_complexity_gamma_pcsi", "time_complexity_gamma_scsi"
#     }

#     if plot_option in [1, 2, 3, 4, 5]:
#         param = {1: 'Ptmax', 2: 'N', 3: 'PRmax', 4: 'a', 5: 'Pcn'}[plot_option]
#         x_type = {1: 'power', 2: 'ris', 3: 'PRmax', 4: 'a', 5: 'Pcn'}[plot_option]
#         x_val = results[param][0]

#         plot_type = input(f"Do you want to plot both SSR and SEE Vs {param}? (yes/no): ").strip().lower() == 'yes'
#         combined_plot = input("Do you want combined plot with separate axes? (yes/no): ").strip().lower() == 'yes'

#         loc, plot_type, smooth, window_length, polyorder, bits_range, quantized_plot, combined_plot = get_plot_config()

#         plot_data(results, x_val, keys_to_average,  loc, window_length, polyorder,plot_type,  x_type, combined_plot, bits_range, quantized_plot, smooth)

#     elif plot_option == 6:
#         param1 = input("Enter the first parameter to vary (e.g., N): ").strip()
#         param2 = input("Enter the second parameter to vary (e.g., Pcn): ").strip()
#         x_type = param2  # Use the second parameter as x_type
#         x_val = results[param2][0]

#         loc, plot_type, smooth, window_length, polyorder, bits_range, quantized_plot, combined_plot = get_plot_config()

#         for p1_value in results[param1]:
#             sub_results = {k: v for k, v in results.items() if k.startswith(f'{param1}_{p1_value}')}

#             plot_data(sub_results, x_val, keys_to_average, loc, window_length, polyorder, plot_type,  x_type, combined_plot, bits_range, quantized_plot, smooth)
    
#     plt.ioff()
#     plt.show()

# if __name__ == "__main__":
    
#     from utils import Utils
    
#     config =  SystemConfig()
    
#     main()
