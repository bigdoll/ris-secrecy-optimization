import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from config import SystemConfig

class Plotter:
    def __init__(self,
                 x_val,
                 data_series,
                 x_type='power',
                 loc='upper left',
                 plot_type='Data Rate',
                 smooth=False,
                 window_length=5,
                 polyorder=1,
                 combined_plot=False):
        """
        x_val: either
            - 1D array (as before), or
            - dict of 1D arrays, { key: x_array, ... }
        data_series: list of dicts, each with:
            - 'key':    must match x_val[key] when x_val is dict
            - 'data':   y-array
            - 'label', 'color', 'marker', 'type', etc.
        """
        # support both single-array and dict-of-arrays for x
        if isinstance(x_val, dict):
            self.x_val_dict = x_val
            self.x_val = None
        else:
            self.x_val_dict = None
            self.x_val = x_val

        self.data_series = data_series
        self.x_type = x_type
        self.loc = loc
        self.plot_type = plot_type
        self.smooth = smooth
        self.window_length = window_length
        self.polyorder = polyorder
        self.combined_plot = combined_plot

    def plot_results(self, save_path=None, formats=['eps', 'png', 'pdf'], fig_name='figure'):
        # set up figure(s)
        if self.combined_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
            self._set_labels(ax1, ax2)
            for series in self.data_series:
                y = self._apply_smoothing(series['data']) if self.smooth else series['data']
                ax = ax1 if 'pCSI' in series.get('type','') else ax2
                self._plot_data(ax, series, y)
            self._finalize_plot(ax1, ax2)
        else:
            fig = plt.figure(figsize=(10, 6))
            self._set_labels()
            for series in self.data_series:
                y = self._apply_smoothing(series['data']) if self.smooth else series['data']
                self._plot_data(plt, series, y)
            self._finalize_plot()

        # save if requested
        if save_path:
            for fmt in formats:
                fig.savefig(f'{save_path}/{fig_name}.{fmt}', format=fmt, transparent=True)

        plt.show(block=True)

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
        return {
            'Ptmax': "Max. available Transmit Power - Ptmax (dBm)",
            'N':      "Number of RIS elements (N)",
            'PRmax':  "Maximum RF Power of the RIS - PRmax (dBm)",
            'a':      "Maximum Gain of the RIS - a (dB)",
            'Pcn':    "Static Power Consumption per RIS element (dBm)",
            'NEV':    "Normalized Error Variance for Eve - NEV (dB)",
            'Rate':   "Data Rate (bps/Hz)",
        }.get(self.x_type, "Parameter")

    def _get_ylabel(self):
        return {
            'Data Rate':         "Data Rate (bps/Hz)",
            'Energy Efficiency': "Energy Efficiency (bits/J)",
        }.get(self.plot_type, "")

    def _apply_smoothing(self, data):
        if self.window_length % 2 == 0:
            self.window_length += 1
        return savgol_filter(data, self.window_length, self.polyorder)

    def _plot_data(self, ax, series, y_data):
        # pick the right x
        if self.x_val_dict is not None:
            key = series.get('key')
            if key not in self.x_val_dict:
                raise KeyError(f"Series key '{key}' not found in x_val dict")
            x_data = self.x_val_dict[key]
        else:
            x_data = self.x_val

        # choose method
        methods = {
            'Data Rate':         ax.plot,
            'Energy Efficiency': ax.plot,
            'Bar':               ax.bar,
            'Scatter':           ax.scatter,
        }
        plot_fn = methods.get(self.plot_type, ax.plot)

        # call it
        if self.plot_type in ('Bar', 'Scatter'):
            plot_fn(x_data, y_data, label=series['label'], color=series['color'])
        else:
            plot_fn(x_data,
                    y_data,
                    label=series['label'],
                    color=series['color'],
                    marker=series['marker'],
                    markersize=14,
                    linewidth=series.get('line_width', 7))

    def _finalize_plot(self, ax1=None, ax2=None):
        if self.combined_plot:
            for ax in (ax1, ax2):
                ax.tick_params(axis='both', labelsize=15)
                ax.legend(fontsize=15, loc=self.loc)
        else:
            plt.tick_params(axis='both', labelsize=15)
            plt.xticks(fontweight='bold')
            plt.yticks(fontweight='bold')
            lg = plt.legend(fontsize=15, loc=self.loc)
            lg.get_frame().set_facecolor('white')
            
# class Plotter:
#     def __init__(self, x_val, data_series, x_type='power', loc='upper left', plot_type='Data Rate', 
#                  smooth=False, window_length=5, polyorder=1, combined_plot=False): # upper left, lower left,  lower right
#         self.x_val = x_val
#         self.data_series = data_series
#         self.x_type = x_type
#         self.loc = loc
#         self.plot_type = plot_type
#         self.smooth = smooth
#         self.window_length = window_length
#         self.polyorder = polyorder
#         self.combined_plot = combined_plot
#         self.save_path = 'data/figures'
#         self.format =  ['eps', 'png', 'pdf']
    
#     def plot_results(self, save_path=None, formats=['eps', 'png', 'pdf'], fig_name='figure'):
#         if self.combined_plot:
#             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
#             self._set_labels(ax1, ax2) #(14, 6)

#             for series in self.data_series:
#                 y_data = self._apply_smoothing(series['data']) if self.smooth else series['data']
#                 self._plot_data(ax1 if 'pCSI' in series['type'] else ax2, series, y_data) # series['label']
            
#             self._finalize_plot(ax1, ax2)
#         else:
#             fig = plt.figure(figsize=(10, 6))
#             self._set_labels()

#             for series in self.data_series:
#                 y_data = self._apply_smoothing(series['data']) if self.smooth else series['data']
#                 self._plot_data(plt, series, y_data)
            
#             self._finalize_plot()
        
#         if save_path:
#             for fmt in formats:
#                 fig.savefig(f'{save_path}/{fig_name}.{fmt}', format=fmt, transparent=True)
                
#         plt.show(block=True)

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
#         # return "Maximum available Transmit Power (dBm)" if self.x_type == 'power' else "Number of RIS elements (N)" #"Maximum Gain (dB) of the RIS"
#         if self.x_type == 'Ptmax':
#             return "Max. available Transmit Power - Ptmax (dBm)"
#         elif self.x_type == 'N':
#             return "Number of RIS elements (N)"
#         elif self.x_type == 'PRmax':
#             return "Maximum RF Power of the RIS - PRmax (dBm)"
#         elif self.x_type == 'a':
#             return "Maximum Gain of the RIS -  a (dB)"
#         elif self.x_type == 'Pcn':
#             return "Static Power Consumption per RIS element (dBm)"
#         elif self.x_type == 'NEV':
#             return "Normalized Error Variance for Eve - NEV (dB)"
#         elif self.x_type == 'Rate':
#             return "Data Rate (bps/Hz)"
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
#                         marker=series['marker'], markersize=14, linewidth=series.get('line_width', 7)) # markersize:8, linewidth:3

#     def _finalize_plot(self, ax1=None, ax2=None):
#         if self.combined_plot:
#             ax1.tick_params(axis='both', labelsize=15)
#             ax2.tick_params(axis='both', labelsize=15)
#             ax1.legend(fontsize=15, loc=self.loc)
#             ax2.legend(fontsize=15, loc=self.loc) # 'upper right'
#         else:
#             plt.tick_params(axis='both', labelsize=15)
#             plt.xticks(fontweight='bold')
#             plt.yticks(fontweight='bold')
#             legend = plt.legend(fontsize=15, loc=self.loc)
#             legend.get_frame().set_facecolor('white')


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


# Example Usage
if __name__ == "__main__":
    
    from utils import Utils
    
    # Enable interactive mode for matplotlib
    plt.ion()
    
    # Initialize system configuration
    config = SystemConfig()
    
    # Keys
    keys_to_average = {
    "sr_uniform_Bob_pcsi", "sr_uniform_Bob_scsi", "sr_uniform_Eve_pcsi",
    "sr_uniform_Eve_scsi", "ssr_uniform_pcsi", "ssr_uniform_scsi", "gee_uniform_Bob_pcsi", "gee_uniform_Bob_scsi",
    "gee_uniform_Eve_pcsi", "gee_uniform_Eve_scsi", "see_uniform_pcsi", "see_uniform_scsi", "ssr_sol_pcsi", "ssr_sol_Q_pcsi",
    "ssr_sol_scsi", "ssr_sol_Q_scsi", "see_sol_pcsi", "see_sol_Q_pcsi", "see_sol_scsi", "see_sol_Q_scsi", "iteration_altopt_pcsi", "iteration_altopt_scsi",
    "iteration_p_pcsi", "iteration_p_scsi", "iteration_gamma_pcsi", "iteration_gamma_scsi",
    "time_complexity_altopt_pcsi", "time_complexity_altopt_scsi", "time_complexity_p_pcsi",
    "time_complexity_p_scsi", "time_complexity_gamma_pcsi", "time_complexity_gamma_scsi", 'sr_sol_Bob_pcsi', 'sr_sol_Eve_pcsi',
    'sr_sol_Bob_scsi', 'sr_sol_Eve_scsi', 'gee_sol_Bob_pcsi', 'gee_sol_Eve_pcsi', 'gee_sol_Bob_scsi', 'gee_sol_Eve_scsi'
    }
    
    # Load the results
    
    # # 1. Varying Ptmax
    results_sr_active = np.load('data/outputs/output_results_algo1_test3_sr_active_2s_100ris_3.0dB_5dBm_0dBvar_Ptmax.npz', allow_pickle=True)['arr_0'].item()  
    results_ee_active = np.load('data/outputs/output_results_algo1_test3_ee_active_2s_100ris_3.0dB_5dBm_0dBvar_Ptmax.npz', allow_pickle=True)['arr_0'].item()  
    
    # results_avg_sr_active = Utils.average_results(results_sr_active, keys_to_average)
    # results_avg_ee_active = Utils.average_results(results_ee_active, keys_to_average)
    
    ## Load the channels:
    config = SystemConfig()
    
    N_range = config.N # config.get('N', [config.N])
    channel_samples_range = Utils.monte_carlo_simulation(
        config.NUM_SAMPLES, config.FILENAME_CHANNEL, N_range, config.r_cell, config.h_BS, config.h_RIS, config.hmin_UE, config.hmax_UE, config.d0,  config.K, config.NR, config.f0, config.lambda_0, config.d, config.c, config.n, config.R_K, config.NEEV, config.vic_percent_eve, channel_model="rician", shadowing=False
    )

    ris_samples_range = Utils.generate_ris_coefficients(
        config.NUM_SAMPLES, config.FILENAME_RIS, N_range, channel_samples_range, (config.PTMIN / config.K) * np.ones(config.K),  config.sigma_sq, config.sigma_RIS_sq, scsi_bool=0, algo="algo1", state=config.ris_state
    )
    
    for sample_index in range(len(channel_samples_range)):
        channel_samples = channel_samples_range[sample_index] # G_B_lst, gE_hat_lst, gE_error_lst, H_lst, sigma_e_sq
        ris_samples = ris_samples_range[sample_index]
        
        Pc_active = config.compute_static_power_consumption(config.N, state='active') 
        Pc_passive = config.compute_static_power_consumption(config.N)
        Pc = Pc_active if config.ris_state == 'active' else Pc_passive

        G_B = channel_samples[config.N]['G_B']
        gE_hat = channel_samples[config.N]['gE']
        gE_error = channel_samples[config.N]['gE_error']
        H = channel_samples[config.N]['H']
        sigma_e_sq = channel_samples[config.N]['sigma_e_sq']
        gE = gE_hat + gE_error[0]
        # gamma = ris_samples[config.N]['gamma']
        
        for power_index, power_value in enumerate(results_sr_active[sample_index][('Ptmax',)]):
            gamma_sr_pcsi = results_sr_active[sample_index]['gamma_sol_pcsi'][power_index]
            gamma_sr_scsi = results_sr_active[sample_index]['gamma_sol_scsi'][power_index]
            gamma_ee_pcsi = results_ee_active[sample_index]['gamma_sol_pcsi'][power_index]
            gamma_ee_scsi = results_ee_active[sample_index]['gamma_sol_scsi'][power_index]
            
            # opt power
            p_sr_pcsi = results_sr_active[sample_index]['p_sol_pcsi'][power_index]
            p_sr_scsi = results_sr_active[sample_index]['p_sol_scsi'][power_index]
            p_ee_pcsi = results_ee_active[sample_index]['p_sol_pcsi'][power_index]
            p_ee_scsi = results_ee_active[sample_index]['p_sol_scsi'][power_index]
            

            # p_uniform = (config.Ptmax / config.K) * np.ones(config.K)
            # R = Utils.compute_R(H, p_uniform, config.sigma_RIS_sq)
            # assert np.all(np.linalg.eigvals(R) > 0), "Matrix R is not positive definite."
            
            # if config.ris_state == 'active':
            #     PRmax = (config.a - 1) * np.real(np.trace(R)) if config.rf_state == 'RF-Gain' else config.PRmax
            # else:
            #     PRmax = 0
            
            # rho_norm = np.sqrt((PRmax + np.real(np.trace(R))) / np.real(np.sum(gamma.conj().T @ R @ gamma))) if config.cons_state == 'global' else 1
            # gamma_random = rho_norm * gamma

            sr_sol_Bob_pcsi = Utils.SR_active_algo1(G_B, H, gamma_sr_pcsi, p_sr_pcsi, config.sigma_sq, config.sigma_RIS_sq, sigma_e_sq, scsi_bool=0, orig_bool=True, Rx="Bob")
            sr_sol_Eve_pcsi = Utils.SR_active_algo1(gE, H, gamma_sr_pcsi, p_sr_pcsi, config.sigma_sq, config.sigma_RIS_sq, sigma_e_sq, scsi_bool=0, orig_bool=True, Rx="Eve")
            ssr_sol_pcsi = max(sr_sol_Bob_pcsi - sr_sol_Eve_pcsi, 0)

            sr_sol_Bob_scsi = Utils.SR_active_algo1(G_B, H, gamma_sr_scsi, p_sr_scsi, config.sigma_sq, config.sigma_RIS_sq, sigma_e_sq, scsi_bool=0, orig_bool=True, Rx="Bob")
            sr_sol_Eve_scsi = Utils.SR_active_algo1(gE, H, gamma_sr_scsi, p_sr_scsi, config.sigma_sq, config.sigma_RIS_sq, sigma_e_sq, scsi_bool=0, orig_bool=True, Rx="Eve")
            ssr_sol_scsi = max(sr_sol_Bob_scsi - sr_sol_Eve_scsi, 0)

            gee_sol_Bob_pcsi = config.BW * Utils.GEE_active_algo1(G_B, H, gamma_ee_pcsi, p_ee_pcsi, config.mu, Pc, config.sigma_sq, config.sigma_RIS_sq, sigma_e_sq, config.ris_state, scsi_bool=0, orig_bool=True, Rx="Bob")
            gee_sol_Eve_pcsi = config.BW * Utils.GEE_active_algo1(gE, H, gamma_ee_pcsi, p_ee_pcsi, config.mu, Pc, config.sigma_sq, config.sigma_RIS_sq, sigma_e_sq, config.ris_state, scsi_bool=0, orig_bool=True, Rx="Eve")
            see_sol_pcsi = max(gee_sol_Bob_pcsi - gee_sol_Eve_pcsi, 0)

            gee_sol_Bob_scsi = config.BW * Utils.GEE_active_algo1(G_B, H, gamma_ee_scsi, p_ee_scsi, config.mu, Pc, config.sigma_sq, config.sigma_RIS_sq, sigma_e_sq, config.ris_state, scsi_bool=0, orig_bool=True, Rx="Bob")
            gee_sol_Eve_scsi = config.BW * Utils.GEE_active_algo1(gE, H, gamma_ee_scsi, p_ee_scsi, config.mu, Pc, config.sigma_sq, config.sigma_RIS_sq, sigma_e_sq, config.ris_state, scsi_bool=0, orig_bool=True, Rx="Eve")
            see_sol_scsi = max(gee_sol_Bob_scsi - gee_sol_Eve_scsi, 0)
            
            
            # For secrecy‐rate dict
            d_sr = results_sr_active[sample_index]
            d_sr.setdefault('sr_sol_Bob_pcsi', []).append(sr_sol_Bob_pcsi)
            d_sr.setdefault('sr_sol_Eve_pcsi', []).append(sr_sol_Eve_pcsi)
            d_sr.setdefault('sr_sol_Bob_scsi', []).append(sr_sol_Bob_scsi)
            d_sr.setdefault('sr_sol_Eve_scsi', []).append(sr_sol_Eve_scsi)

            # For secrecy‐EE dict
            d_ee = results_ee_active[sample_index]
            d_ee.setdefault('gee_sol_Bob_pcsi', []).append(gee_sol_Bob_pcsi)
            d_ee.setdefault('gee_sol_Eve_pcsi', []).append(gee_sol_Eve_pcsi)
            d_ee.setdefault('gee_sol_Bob_scsi', []).append(gee_sol_Bob_scsi)
            d_ee.setdefault('gee_sol_Eve_scsi', []).append(gee_sol_Eve_scsi)
    
            # results_sr_active[sample_index]['sr_sol_Bob_pcsi'].append(sr_sol_Bob_pcsi)
            # results_sr_active[sample_index]['sr_sol_Eve_pcsi'].append(sr_sol_Eve_pcsi)
            # results_sr_active[sample_index]['sr_sol_Bob_scsi'].append(sr_sol_Bob_scsi)
            # results_sr_active[sample_index]['sr_sol_Eve_scsi'].append(sr_sol_Eve_scsi)
            # results_ee_active[sample_index]['gee_sol_Bob_pcsi'].append(gee_sol_Bob_pcsi)
            # results_ee_active[sample_index]['gee_sol_Eve_pcsi'].append(gee_sol_Eve_pcsi)
            # results_ee_active[sample_index]['gee_sol_Bob_scsi'].append(gee_sol_Bob_scsi)
            # results_ee_active[sample_index]['gee_sol_Eve_scsi'].append(gee_sol_Eve_scsi)

     
    ### SEE Vs. SSR Plot:
    
    results_avg_sr_active = Utils.average_results(results_sr_active, keys_to_average)
    results_avg_ee_active = Utils.average_results(results_ee_active, keys_to_average)
    
    x_vals = {
    'pCSI': results_avg_sr_active['sr_sol_Bob_pcsi'],
    'sCSI': results_avg_sr_active['sr_sol_Bob_scsi'],
    # add more if you like (e.g. active RIS vs. passive RIS)
    }
       
    # x_vals = {
    # 'pCSI': results_avg_sr_active['ssr_sol_pcsi'],
    # 'sCSI': results_avg_sr_active['ssr_sol_scsi'],
    # # add more if you like (e.g. active RIS vs. passive RIS)
    # }
    
    see_series = [
        {
        'key':   'pCSI',
        'data':  results_avg_ee_active['gee_sol_Bob_pcsi'],
        'label': 'GEE (Alg 3, pCSI)',
        'color': 'blue',
        'marker':'o',
        'type':  'pCSI'
        },
        {
        'key':   'sCSI',
        'data':  results_avg_ee_active['gee_sol_Bob_scsi'],
        'label': 'GEE (Alg 6, sCSI)',
        'color': 'red',
        'marker':'x',
        'type':  'sCSI'
        },
    ]
     
    # see_series = [
    #     {
    #     'key':   'pCSI',
    #     'data':  results_avg_ee_active['see_sol_pcsi'],
    #     'label': 'SEE (Alg 3, pCSI)',
    #     'color': 'blue',
    #     'marker':'o',
    #     'type':  'pCSI'
    #     },
    #     {
    #     'key':   'sCSI',
    #     'data':  results_avg_ee_active['see_sol_scsi'],
    #     'label': 'SEE (Alg 6, sCSI)',
    #     'color': 'red',
    #     'marker':'x',
    #     'type':  'sCSI'
    #     },
    # ]

    plotter = Plotter(
    x_val=x_vals,
    data_series=see_series,
    x_type='Rate',            # since x-axis is SSR (Rate)
    plot_type='Energy Efficiency',
    combined_plot=False
    )
    plotter.plot_results()

    
    # # # Extract data for plotting
    # x_val = np.arange(-20, 52, 2) #config.power_range_dbm #results['Ptmax'][0]
    
    # results_avg_sr = Utils.average_values_by_key_equal_length(results_sr, config.NUM_SAMPLES)
    # results_avg_ee = Utils.average_values_by_key_equal_length(results_ee, config.NUM_SAMPLES)   
    
    # # 2. varying N
    # # results_sr_active = np.load('data/output_N/output_results_algo1_sr_active_2s_30.0dBm_10.0dB_5dBm_N.npz', allow_pickle=True)['arr_0'].item()
    # results_ee_active = np.load('data/outputs/output_results_algo1_ee_active_2s_0.0dBm_3.0dB_5dBm_0dBvar_N.npz', allow_pickle=True)['arr_0'].item()
    # # results_sr_passive = np.load('data/output_N/output_results_algo1_sr_passive_2s_30.0dBm_0.0dB_0dBm_N.npz', allow_pickle=True)['arr_0'].item()
    # results_ee_passive = np.load('data/outputs/output_results_algo1_ee_passive_2s_0.0dBm_0.0dB_0dBm_0dBvar_N.npz', allow_pickle=True)['arr_0'].item()
    
    # # Averaging results over num_samples
    # # results_avg_sr_active =Utils.average_results(results_sr_active, keys_to_average)
    # results_avg_ee_active =Utils.average_results(results_ee_active, keys_to_average)
    # # results_avg_sr_passive =Utils.average_results(results_sr_passive, keys_to_average)
    # results_avg_ee_passive =Utils.average_results(results_ee_passive, keys_to_average)
    
    # # Extract data for plotting - 
    # x_val = np.arange(10,220,10) #results['Ptmax'][0]
    
    # # 3. Varying PRmax
    # results_sr_active = np.load('data/outputs/output_results_algo1_sr_active_2s_100ris_0.0dBm_5dBm_0dBvar_PRmax.npz', allow_pickle=True)['arr_0'].item()  
    # results_sr_passive = np.load('data/outputs/output_results_algo1_sr_passive_2s_100ris_0.0dBm_0dBm_0dBvar_PRmax.npz', allow_pickle=True)['arr_0'].item()  
    # results_ee_active = np.load('data/outputs/output_results_algo1_ee_active_2s_100ris_0.0dBm_5dBm_0dBvar_PRmax.npz', allow_pickle=True)['arr_0'].item()  
    # results_ee_passive = np.load('data/outputs/output_results_algo1_ee_passive_2s_100ris_0.0dBm_0dBm_0dBvar_PRmax.npz', allow_pickle=True)['arr_0'].item()  
    
    # results_avg_sr_active = Utils.average_results(results_sr_active, keys_to_average)
    # results_avg_sr_passive = Utils.average_results(results_sr_passive, keys_to_average)
    # results_avg_ee_active = Utils.average_results(results_ee_active, keys_to_average)
    # results_avg_ee_passive = Utils.average_results(results_ee_passive, keys_to_average)
    
    # # # Extract data for plotting
    # x_val = np.arange(-30, 32, 2) #config.power_range_dbm #results['PRmax'][0] 
    
    # #  4. Varying Pcn for each N
    # 'data/outputs/output_results_algo1_ee_active_2s_100ris_3.0dB_30.0dBm_0dBvar_Pcn.npz'
    # 'data/outputs/output_results_algo1_ee_active_2s_200ris_3.0dB_30.0dBm_0dBvar_Pcn.npz'
    # 'data/outputs/output_results_algo1_ee_passive_2s_100ris_0.0dB_30.0dBm_0dBvar_Pcn.npz'
    # 'data/outputs/output_results_algo1_ee_passive_2s_200ris_0.0dB_30.0dBm_0dBvar_Pcn.npz'
    # 'data/output_test/output_results_algo1_ee_active_2s_100ris_5.0dB_30.0dBm_0dBvar_Pcn.npz'
    # results_ee_N100_active = np.load('data/output_test/output_results_algo1_ee_active_2s_100ris_5.0dB_30.0dBm_0dBvar_Pcn.npz', allow_pickle=True)['arr_0'].item() 
    # results_ee_N200_active = np.load('data/output_test/output_results_algo1_ee_active_2s_200ris_5.0dB_30.0dBm_0dBvar_Pcn.npz', allow_pickle=True)['arr_0'].item() 
    # results_ee_N100_passive = np.load('data/output_test/output_results_algo1_ee_passive_2s_100ris_0.0dB_30.0dBm_0dBvar_Pcn.npz', allow_pickle=True)['arr_0'].item() 
    # results_ee_N200_passive = np.load('data/output_test/output_results_algo1_ee_passive_2s_200ris_0.0dB_30.0dBm_0dBvar_Pcn.npz', allow_pickle=True)['arr_0'].item() 
   
    # results_avg_ee_N100_active = Utils.average_results(results_ee_N100_active, keys_to_average)
    # results_avg_ee_N200_active = Utils.average_results(results_ee_N200_active, keys_to_average)
    # results_avg_ee_N100_passive = Utils.average_results(results_ee_N100_passive, keys_to_average)
    # results_avg_ee_N200_passive = Utils.average_results(results_ee_N200_passive, keys_to_average)
    
    # x_val = np.arange(0, 32, 2)
    
    # # 5. Varying a 
    # data/output_a/output_results_algo1_ee_active_2s_100ris_30.0dBm_5dBm_a
    # data/outputs/output_results_algo1_ee_active_2s_100ris_0.0dBm_5dBm_0dBvar_a.npz
    # 'data/output_a/output_results_algo1_ee_passive_2s_100ris_30.0dBm_0dBm_a.npz'
    
    # results_ee_active = np.load('data/output_test/output_results_algo1_ee_active_2s_100ris_10.0dBm_5dBm_0dBvar_a.npz', allow_pickle=True)['arr_0'].item() 
    # results_ee_passive = np.load('data/output_test/output_results_algo1_ee_passive_2s_100ris_10.0dBm_0dBm_0dBvar_a.npz', allow_pickle=True)['arr_0'].item() 
    
   
    # results_avg_ee_active = Utils.average_results(results_ee_active, keys_to_average)
    # results_avg_ee_passive = Utils.average_results(results_ee_passive, keys_to_average)
   
    # x_val = np.arange(0,32,2)
      
    # # 6. Varying NEV (Normalized Error Variance) - Eve
    # 'data/outputs/output_results_algo1_ee_active_2s_100ris_30.0dBm_5dBm_3.0_NEEV.npz'
    # 'data/outputs/output_results_algo1_ee_active_2s_200ris_30.0dBm_5dBm_3.0_NEEV.npz'
    # 'data/output_test/output_results_algo1_ee_active_2s_100ris_30.0dBm_5dBm_5.0_NEEV.npz'
    # 'data/output_test/output_results_algo1_ee_active_2s_200ris_30.0dBm_5dBm_5.0_NEEV.npz'
    # results_ee_N100_active = np.load('data/outputs/output_results_algo1_ee_active_2s_100ris_30.0dBm_5dBm_3.0_NEEV.npz', allow_pickle=True)['arr_0'].item() 
    # results_ee_N200_active = np.load('data/outputs/output_results_algo1_ee_active_2s_200ris_30.0dBm_5dBm_3.0_NEEV.npz', allow_pickle=True)['arr_0'].item() 
    # results_ee_N100_passive = np.load('data/output_test/output_results_algo1_ee_passive_2s_100ris_30.0dBm_0dBm_0.0_NEEV.npz', allow_pickle=True)['arr_0'].item() 
    # results_ee_N200_passive = np.load('data/output_test/output_results_algo1_ee_passive_2s_200ris_30.0dBm_0dBm_0.0_NEEV.npz', allow_pickle=True)['arr_0'].item() 
    
   
    # results_avg_ee_N100_active = Utils.average_results(results_ee_N100_active, keys_to_average)
    # results_avg_ee_N200_active = Utils.average_results(results_ee_N200_active, keys_to_average)
    # results_avg_ee_N100_passive = Utils.average_results(results_ee_N100_passive, keys_to_average)
    # results_avg_ee_N200_passive = Utils.average_results(results_ee_N200_passive, keys_to_average)
   
    # x_val = np.arange(-30,21,1)
    # for i in range(5):
    #     temp = results_avg_sr_active['ssr_sol_scsi'][35-i]
    #     results_avg_sr_active['ssr_sol_Q_scsi'][(2,2)][35-i] = temp
    
    # if config.quantization:
    #     colors = ['cyan', 'green', 'blue', 'magenta']
    #     markers = ['x', '*', '^', 's']
        
    #     rate_series_Q_pcsi = [
    #         {'data': results_avg_sr_active['ssr_sol_pcsi'], 'label': 'SSR Maximization by Alg. 3 - No Quantization', 'color': 'red', 'marker': 'o', 'type': 'pCSI'}
    #     ] + [{'data': results_avg_sr_active['ssr_sol_Q_pcsi'][bits], 'label': f'SSR Maximization by Alg. 3 - {bits} bits', 'color': colors[i % len(colors)], 'marker': markers[i % len(markers)], 'type': 'pCSI'} for i, bits in enumerate(config.bits_range)]
        
    #     rate_series_Q_scsi = [
    #         {'data': results_avg_sr_active['ssr_sol_scsi'], 'label': 'SSR Maximization by Alg. 6 - No Quantization', 'color': 'red', 'marker': 'o', 'type': 'sCSI'}
    #     ] + [{'data': results_avg_sr_active['ssr_sol_Q_scsi'][bits], 'label': f'SSR Maximization by Alg. 6 - {bits} bits', 'color': colors[i % len(colors)], 'marker': markers[i % len(markers)], 'type': 'sCSI'} for i, bits in enumerate(config.bits_range)]
        
    #     energy_series_Q_pcsi = [
    #         {'data': results_avg_ee_active['see_sol_pcsi'], 'label': 'SEE Maximization by Alg. 3 - No Quantization', 'color': 'red', 'marker': 'o', 'type': 'pCSI'}
    #     ] + [{'data': results_avg_ee_active['see_sol_Q_pcsi'][bits], 'label': f'SEE Maximization by Alg. 3 - {bits} bits', 'color': colors[i % len(colors)], 'marker': markers[i % len(markers)], 'type': 'pCSI'} for i, bits in enumerate(config.bits_range)]
        
    #     energy_series_Q_scsi = [
    #         {'data': results_avg_ee_active['see_sol_scsi'], 'label': 'SEE Maximization by Alg. 6 - No Quantization', 'color': 'red', 'marker': 'o', 'type': 'sCSI'}
    #     ] + [{'data': results_avg_ee_active['see_sol_Q_scsi'][bits], 'label': f'SEE Maximization by Alg. 6 - {bits} bits', 'color': colors[i % len(colors)], 'marker': markers[i % len(markers)], 'type': 'sCSI'} for i, bits in enumerate(config.bits_range)]
        
    #     # Combined plot option
    #     combined_plot = input("Do you want to plot combined figures with separate axes - with Quantization? (yes/no): ").strip().lower() == 'yes'

    #     # Plot results
    #     if combined_plot:
    #         # Plotter - Ptmax
    #         plotter_rate_Q_combined = Plotter(x_val, rate_series_Q_pcsi + rate_series_Q_scsi, x_type='Ptmax', plot_type='Data Rate', smooth=False, combined_plot=True) # x_type='Ptmax',
    #         plotter_energy_Q_combined = Plotter(x_val, energy_series_Q_pcsi + energy_series_Q_scsi, x_type='Ptmax', plot_type='Energy Efficiency', smooth=False, combined_plot=True) # x_type='Ptmax',
            
    #         # plot_results - Ptmax
    #         plotter_rate_Q_combined.plot_results(save_path=None, formats=['eps', 'png', 'pdf'], fig_name='figure_SSR_Q_Vs_Ptmax_0dB') # save_path='data/figures/SSR_Q_figs'
    #         plotter_energy_Q_combined.plot_results(save_path=None, formats=['eps', 'png', 'pdf'], fig_name='figure_SEE_Q_Vs_Ptmax_0dB') # save_path='data/figures/SEE_Q_figs'
    #     else:
    #         # Plotter - Ptmax
    #         plotter_rate_Q_pcsi = Plotter(x_val, rate_series_Q_pcsi, x_type='Ptmax', plot_type='Data Rate', smooth=True, combined_plot=False) # x_type='Ptmax',
    #         plotter_rate_Q_scsi = Plotter(x_val, rate_series_Q_scsi, x_type='Ptmax', plot_type='Data Rate', smooth=True, combined_plot=False)
    #         plotter_energy_Q_pcsi = Plotter(x_val, energy_series_Q_pcsi, x_type='Ptmax', plot_type='Energy Efficiency', smooth=True, combined_plot=False) # x_type='Ptmax', x_type='N'
    #         plotter_energy_Q_scsi = Plotter(x_val, energy_series_Q_scsi, x_type='Ptmax', plot_type='Energy Efficiency', smooth=True, combined_plot=False)
            
    #         # plot_results - Ptmax
    #         plotter_rate_Q_pcsi.plot_results(save_path='data/figures/TIF_figs', formats=['eps', 'png', 'pdf'], fig_name='figure_SSR_Q_Vs_Ptmax_pCSI')
    #         plotter_rate_Q_scsi.plot_results(save_path='data/figures/TIF_figs', formats=['eps', 'png', 'pdf'], fig_name='figure_SSR_Q_Vs_Ptmax_sCSI')
    #         plotter_energy_Q_pcsi.plot_results(save_path='data/figures/TIF_figs', formats=['eps', 'png', 'pdf'], fig_name='figure_SEE_Q_Vs_Ptmax_pCSI')
    #         plotter_energy_Q_scsi.plot_results(save_path='data/figures/TIF_figs', formats=['eps', 'png', 'pdf'], fig_name='figure_SEE_Q_Vs_Ptmax_sCSI')
    
    # # Rate and EE series for Ptmax     
    # rate_series = [
    #     {'data': results_avg_sr_active['ssr_sol_pcsi'], 'label': 'SSR by Algorithm 3 for SSR maximization', 'color': 'blue', 'marker': 'o', 'type': 'pCSI'},
    #     {'data': results_avg_sr_active['ssr_sol_scsi'], 'label': 'SSR by Algorithm 6 for SSR maximization', 'color': 'red', 'marker': 'x', 'type': 'sCSI'},
    #     # {'data': results_avg_ee_active['ssr_sol_pcsi'], 'label': 'SSR by Algorithm 3 for SEE maximization', 'color': 'green', 'marker': '^', 'type': 'pCSI'},
    #     # {'data': results_avg_ee_active['ssr_sol_scsi'], 'label': 'SSR by Algorithm 6 for SEE maximization', 'color': 'magenta', 'marker': '*', 'type': 'sCSI'},
    #     # {'data': results_avg_sr_active['ssr_uniform_pcsi'], 'label': 'SSR by random RIS phases, uniform \nTx powers and RIS moduli', 'color': 'black', 'marker': 's', 'type': 'pCSI'},
    #     # {'data': results_avg_sr_active['ssr_uniform_scsi'], 'label': 'SSR with random phases - sCSI', 'color': 'cyan', 'marker': '+', 'type': 'sCSI'}
    # ]
    
    # energy_series = [
    #     {'data': results_avg_ee_active['see_sol_pcsi'], 'label': 'SEE by Algorithm 3 for SEE maximization', 'color': 'blue', 'marker': 'o', 'type': 'pCSI'},
    #     {'data': results_avg_ee_active['see_sol_scsi'], 'label': 'SEE by Algorithm 6 for SEE maximization', 'color': 'red', 'marker': 'x', 'type': 'sCSI'},
    #     # {'data': results_avg_sr_active['see_sol_pcsi'], 'label': 'SEE by Algorithm 3 for SSR maximization', 'color': 'green', 'marker': '^', 'type': 'pCSI'},
    #     # {'data': results_avg_sr_active['see_sol_scsi'], 'label': 'SEE by Algorithm 6 for SSR maximization', 'color': 'magenta', 'marker': '*', 'type': 'sCSI'},
    #     # {'data': results_avg_ee_active['see_uniform_pcsi'], 'label': 'SEE by random RIS phases, uniform \nTx powers and RIS moduli', 'color': 'black', 'marker': 's', 'type': 'pCSI'},
    #     # {'data': results_avg_ee_active['see_uniform_scsi'], 'label': 'SEE with random phases - sCSI', 'color': 'cyan', 'marker': '+', 'type': 'sCSI'}
    #     ]
    
    # # Energy series for N:
    # rate_series = [
    #     {'data': results_avg_sr_active['ssr_sol_pcsi'], 'label': 'SSR Max. - pCSI', 'color': 'blue', 'marker': 'o', 'type': 'pCSI'},
    #     {'data': results_avg_sr_active['ssr_sol_scsi'], 'label': 'SSR Max. - sCSI', 'color': 'red', 'marker': 'x', 'type': 'sCSI'},
    #     {'data': results_avg_ee_active['ssr_sol_pcsi'], 'label': 'SSR with SEE Max. - pCSI', 'color': 'green', 'marker': 'o', 'type': 'pCSI'},
    #     {'data': results_avg_ee_active['ssr_sol_scsi'], 'label': 'SSR with SEE Max. - sCSI', 'color': 'magenta', 'marker': 'x', 'type': 'sCSI'},
    #     {'data': results_avg_sr_active['ssr_uniform_pcsi'], 'label': 'SSR with random phases - pCSI', 'color': 'black', 'marker': 'o', 'type': 'pCSI'},
    #     {'data': results_avg_sr_active['ssr_uniform_scsi'], 'label': 'SSR with random phases - sCSI', 'color': 'cyan', 'marker': 'x', 'type': 'sCSI'}
    # ]
    
    # energy_series = [
    #     {'data': results_avg_ee_active['see_sol_pcsi'], 'label': 'SEE Max. active - pCSI ', 'color': 'red', 'marker': 'o', 'type': 'pCSI'},
    #     {'data': results_avg_ee_active['see_sol_scsi'], 'label': 'SEE Max. active - sCSI', 'color': 'magenta', 'marker': 'p', 'type': 'sCSI'},
    #     # {'data': results_avg_sr_active['see_sol_pcsi'], 'label': 'SEE with SSR Max. - pCSI', 'color': 'green', 'marker': 'o', 'type': 'pCSI'},
    #     # {'data': results_avg_sr_active['see_sol_scsi'], 'label': 'SEE with SSR Max. - sCSI', 'color': 'magenta', 'marker': 'x', 'type': 'sCSI'},
    #     {'data': results_avg_ee_active['see_uniform_pcsi'], 'label': 'SEE with random phases - pCSI', 'color': 'black', 'marker': '^', 'type': 'pCSI'},
    #     {'data': results_avg_ee_active['see_uniform_scsi'], 'label': 'SEE with random phases - sCSI', 'color': 'cyan', 'marker': 's', 'type': 'sCSI'}
        
    #     # {'data': results_avg_ee_active['see_uniform_scsi'], 'label': 'SEE with random phases - active - sCSI ', 'color': 'black', 'marker': 'x', 'type': 'sCSI'},
    #     # {'data': results_avg_sr_active['see_sol_pcsi'], 'label': 'SEE with SSR Max. active', 'color': 'magenta', 'marker': 'o', 'type': 'pCSI'},
    #     # {'data': results_avg_sr_active['see_sol_scsi'], 'label': 'SEE with SSR Max. active', 'color': 'magenta', 'marker': 'x', 'type': 'sCSI'},
    #     # {'data': results_avg_ee_passive['see_sol_pcsi'], 'label': 'SEE Max. passive - pCSI', 'color': 'blue', 'marker': 'o', 'type': 'pCSI'},
    #     # {'data': results_avg_ee_passive['see_sol_scsi'], 'label': 'SEE Max. passive - sCSI', 'color': 'green', 'marker': 'x', 'type': 'sCSI'},
    #     # {'data': results_avg_ee_passive['see_uniform_pcsi'], 'label': 'SEE with random phases - passive', 'color': 'orange', 'marker': 's', 'type': 'pCSI'},
    #     # {'data': results_avg_ee_passive['see_uniform_scsi'], 'label': 'SEE with random phases - passive - sCSI', 'color': 'yellow', 'marker': '^', 'type': 'sCSI'}
    #     # {'data': results_avg_sr_passive['see_sol_pcsi'], 'label': 'SEE with SSR Max. passive', 'color': 'green', 'marker': '*', 'type': 'pCSI'},
    #     # {'data': results_avg_sr_passive['see_sol_scsi'], 'label': 'SEE with SSR Max. passive', 'color': 'green', 'marker': '^', 'type': 'sCSI'}
    # ]
    
    # # # EE series for Pcn
    # energy_series = [
    #     # {'data': results_avg_ee_N100_active['see_sol_pcsi'], 'label': 'SEE maximization by Alg. 3 - Active. N = 100', 'color': 'magenta', 'marker': 'o', 'type': 'pCSI'},
    #     {'data': results_avg_ee_N100_active['see_sol_scsi'], 'label': 'SEE maximization by Alg. 6 - Active. N = 100', 'color': 'orange', 'marker': 'x', 'type': 'sCSI'},
    #     # {'data': results_avg_ee_N200_active['see_sol_pcsi'], 'label': 'SEE maximization by Alg. 3 - Active. N = 200', 'color': 'red', 'marker': '^', 'type': 'pCSI'},
    #     {'data': results_avg_ee_N200_active['see_sol_scsi'], 'label': 'SEE maximization by Alg. 6 - Active. N = 200', 'color': 'yellow', 'marker': '*', 'type': 'sCSI'},
    #     # {'data': results_avg_ee_N100_passive['see_sol_pcsi']*len(x_val), 'label': 'SEE maximization by Alg. 3 - Passive. N = 100', 'color': 'blue', 'marker': 's', 'type': 'pCSI'},
    #     {'data': results_avg_ee_N100_passive['see_sol_scsi']*len(x_val), 'label': 'SEE maximization by Alg. 6 - Passive. N = 100', 'color': 'cyan', 'marker': 'p', 'type': 'sCSI'},
    #     # {'data': results_avg_ee_N200_passive['see_sol_pcsi']*len(x_val), 'label': 'SEE maximization by Alg. 3 - Passive. N = 200', 'color': 'green', 'marker': 'd', 'type': 'pCSI'},
    #     {'data': results_avg_ee_N200_passive['see_sol_scsi']*len(x_val), 'label': 'SEE maximization by Alg. 6 - Passive. N = 200', 'color': 'black', 'marker': 'h', 'type': 'sCSI'}
    # ]
    
    # # Rate and EE series for PRmax     
    # rate_series = [
    #     {'data': results_avg_sr_active['ssr_sol_pcsi'], 'label': 'SSR Max. active - pCSI', 'color': 'blue', 'marker': 'x', 'type': 'pCSI'},
    #     {'data': results_avg_sr_active['ssr_sol_scsi'], 'label': 'SSR Max. active - sCSI', 'color': 'red', 'marker': 'h', 'type': 'sCSI'},
    #     # {'data': results_avg_sr_passive['ssr_sol_pcsi']*len(x_val), 'label': 'SSR Max. passive - pCSI ', 'color': 'green', 'marker': 'o', 'type': 'pCSI'},
    #     # {'data': results_avg_sr_passive['ssr_sol_scsi']*len(x_val), 'label': 'SSR Max. passive - sCSI', 'color': 'magenta', 'marker': 'p', 'type': 'sCSI'},
    #     {'data': results_avg_ee_active['ssr_sol_pcsi'], 'label': 'SSR with SEE Max. - pCSI', 'color': 'green', 'marker': '^', 'type': 'pCSI'},
    #     {'data': results_avg_ee_active['ssr_sol_scsi'], 'label': 'SSR with SEE Max. - sCSI', 'color': 'magenta', 'marker': 'p', 'type': 'sCSI'},
    #     {'data': results_avg_sr_active['ssr_uniform_pcsi'], 'label': 'SSR with random phases - pCSI', 'color': 'black', 'marker': '*', 'type': 'pCSI'},
    #     {'data': results_avg_sr_active['ssr_uniform_scsi'], 'label': 'SSR with random phases - sCSI', 'color': 'cyan', 'marker': 's', 'type': 'sCSI'}
    # ]
    
    # energy_series = [
    #     {'data': results_avg_ee_active['see_sol_pcsi'], 'label': 'SEE Max. - pCSI', 'color': 'blue', 'marker': 'x', 'type': 'pCSI'},
    #     {'data': results_avg_ee_active['see_sol_scsi'], 'label': 'SEE Max. - sCSI', 'color': 'red', 'marker': 'h', 'type': 'sCSI'},
    #     # {'data': results_avg_ee_passive['see_sol_pcsi']*len(x_val), 'label': 'SEE Max. passive - pCSI ', 'color': 'green', 'marker': 'o', 'type': 'pCSI'},
    #     # {'data': results_avg_ee_passive['see_sol_scsi']*len(x_val), 'label': 'SEE Max. passive - sCSI', 'color': 'magenta', 'marker': 'p', 'type': 'sCSI'},
    #     {'data': results_avg_sr_active['see_sol_pcsi'], 'label': 'SEE with SSR Max. - pCSI', 'color': 'green', 'marker': '^', 'type': 'pCSI'},
    #     {'data': results_avg_sr_active['see_sol_scsi'], 'label': 'SEE with SSR Max. - sCSI', 'color': 'magenta', 'marker': 'p', 'type': 'sCSI'},
    #     {'data': results_avg_ee_active['see_uniform_pcsi'], 'label': 'SEE with random phases - pCSI', 'color': 'black', 'marker': '*', 'type': 'pCSI'},
    #     {'data': results_avg_ee_active['see_uniform_scsi'], 'label': 'SEE with random phases - sCSI', 'color': 'cyan', 'marker': 's', 'type': 'sCSI'}
    #     ]
    
    # # # EE series for a
    # energy_series = [
    #     {'data': results_avg_ee_active['see_sol_pcsi'], 'label': 'SEE Max. active - pCSI', 'color': 'magenta', 'marker': 'o', 'type': 'pCSI'},
    #     {'data': results_avg_ee_active['see_sol_scsi'], 'label': 'SEE Max. active - sCSI', 'color': 'red', 'marker': 'x', 'type': 'sCSI'},
    #     # {'data': results_avg_ee_active['see_uniform_pcsi'], 'label': 'SEE Max. active - pCSI', 'color': 'green', 'marker': '^', 'type': 'pCSI'},
    #     # {'data': results_avg_ee_active['see_uniform_scsi'], 'label': 'SEE Max. active - sCSI', 'color': 'blue', 'marker': 's', 'type': 'sCSI'},
    #     {'data': results_avg_ee_passive['see_sol_pcsi']*len(x_val), 'label': 'SEE Max. passive - pCSI', 'color': 'green', 'marker': '^', 'type': 'pCSI'},
    #     {'data': results_avg_ee_passive['see_sol_scsi']*len(x_val), 'label': 'SEE Max. passive - sCSI', 'color': 'blue', 'marker': 's', 'type': 'sCSI'}
    # ]
    
    # # EE series for NEV
    # energy_series = [
    #     {'data': results_avg_ee_N100_active['see_sol_pcsi'], 'label': 'SEE maximization by Alg. 3 - N = 100', 'color': 'green', 'marker': 'o', 'type': 'pCSI'},
    #     {'data': results_avg_ee_N100_active['see_sol_scsi'], 'label': 'SEE maximization by Alg. 6 - N = 100', 'color': 'magenta', 'marker': 'x', 'type': 'sCSI'},
    #     {'data': results_avg_ee_N200_active['see_sol_pcsi'], 'label': 'SEE maximization by Alg. 3 - N = 200', 'color': 'blue', 'marker': '^', 'type': 'pCSI'},
    #     {'data': results_avg_ee_N200_active['see_sol_scsi'], 'label': 'SEE maximization by Alg. 6 - N = 200', 'color': 'red', 'marker': '*', 'type': 'sCSI'}
    #     # {'data': results_avg_ee_N100_passive['see_sol_pcsi'], 'label': 'SEE Max. passive - N = 100 - pCSI', 'color': 'black', 'marker': '+', 'type': 'pCSI'},
    #     # {'data': results_avg_ee_N100_passive['see_sol_scsi'], 'label': 'SEE Max. passive - N = 100 - sCSI', 'color': 'cyan', 'marker': 'x', 'type': 'sCSI'},
    #     # {'data': results_avg_ee_N200_passive['see_sol_pcsi'], 'label': 'SEE Max. passive - N = 200 - pCSI', 'color': 'yellow', 'marker': 'o', 'type': 'pCSI'},
    #     # {'data': results_avg_ee_N200_passive['see_sol_scsi'], 'label': 'SEE Max. passive - N = 200 - sCSI', 'color': 'orange', 'marker': 's', 'type': 'sCSI'}
    # ]


    # # Combined plot option
    # combined_plot = input("Do you want to plot combined figures with separate axes - without Quantization? (yes/no): ").strip().lower() == 'yes'

    # # Plot results
    # if combined_plot:
    #     # # Plotter - Ptmax
    #     plotter_rate_combined = Plotter(x_val, rate_series, x_type='Ptmax', plot_type='Data Rate', smooth=False, combined_plot=True) # x_type='Ptmax'
    #     plotter_energy_combined = Plotter(x_val, energy_series, x_type='Ptmax', plot_type='Energy Efficiency', smooth=False, combined_plot=True) # x_type='Ptmax'
        
    #     # # Plotter - N
    #     # plotter_rate_combined = Plotter(x_val, rate_series, x_type='N', plot_type='Data Rate', smooth=False, combined_plot=True)
    #     # plotter_energy_combined = Plotter(x_val, energy_series, x_type='N', plot_type='Energy Efficiency', smooth=False, combined_plot=True)
        
    #     # # Plotter - PRmax
    #     # plotter_rate_combined = Plotter(x_val, rate_series, x_type='PRmax', plot_type='Data Rate', smooth=False, combined_plot=True) # x_type='PRmax'
    #     # plotter_energy_combined = Plotter(x_val, energy_series, x_type='PRmax', plot_type='Energy Efficiency', smooth=False, combined_plot=True) # x_type='PRmax'
        
    #     # # Plotter - Pcn
    #     # plotter_energy_combined = Plotter(x_val, energy_series, x_type='Pcn', plot_type='Energy Efficiency', smooth=False, combined_plot=True)
        
    #     # # Plotter - a
    #     # plotter_energy_combined = Plotter(x_val, energy_series, x_type='a', plot_type='Energy Efficiency', smooth=False, combined_plot=True)
        
    #     # # Plotter - NEV
    #     # plotter_rate_combined = Plotter(x_val, rate_series, x_type='NEV', plot_type='Data Rate', smooth=False, combined_plot=True)
    #     # plotter_energy_combined = Plotter(x_val, energy_series, x_type='NEV', plot_type='Energy Efficiency', smooth=False, combined_plot=True)
        
        
    #     # # Plot results - Ptmax
    #     plotter_rate_combined.plot_results(save_path=None, formats=['eps', 'png', 'pdf'], fig_name='figure_SSR_Vs_Ptmax_0dB') # save_path='data/figures/SSR_figs'
    #     plotter_energy_combined.plot_results(save_path=None, formats=['eps', 'png', 'pdf'], fig_name='figure_SEE_Vs_Ptmax_0dB') # save_path='data/figures/SEE_figs'
        
    # #     # # plot_results - N
    # #     # plotter_rate_combined.plot_results(save_path=None, formats=['eps', 'png', 'pdf'], fig_name='figure_SSR_Vs_N') # save_path='data/figures/SSR_figs', None
    # #     # plotter_energy_combined.plot_results(save_path=None, formats=['eps', 'png', 'pdf'], fig_name='figure_SEE_Vs_N_0dB') # 'data/figures/SEE_figs', None
        
    # #     # # Plot results - PRmax
    # #     # plotter_rate_combined.plot_results(save_path=None, formats=['eps', 'png', 'pdf'], fig_name='figure_SSR_Vs_PRmax_0dB') # save_path='data/figures/SSR_figs'
    # #     # plotter_energy_combined.plot_results(save_path=None, formats=['eps', 'png', 'pdf'], fig_name='figure_SEE_Vs_PRmax_0dB') # save_path='data/figures/SEE_figs'
        
    # #     # # plot_results - Pcn
    # #     # 'data/figures/SEE_figs'
    # #     # plotter_energy_combined.plot_results(save_path='data/figures/TIF_figs', formats=['eps', 'png', 'pdf'], fig_name='figure_SEE_Vs_Pcn') #'data/figures/SEE_figs', 'data/output_test/figs/'
        
        
    # #     # # plot_results - a
    # #     # plotter_energy_combined.plot_results(save_path=None, formats=['eps', 'png', 'pdf'], fig_name='figure_SEE_Vs_a') # 'data/figures/SEE_figs'
        
    # #     # # plot_results - NEV
    # #     # plotter_rate_combined.plot_results(save_path=None, formats=['eps', 'png', 'pdf'], fig_name='figure_SSR_Vs_NEV') # save_path='data/figures/SSR_figs', None
    # #     # plotter_energy_combined.plot_results(save_path=None, formats=['eps', 'png', 'pdf'], fig_name='figure_SEE_Vs_NEV') # 'data/figure
        
    # # else:
    #     # # Plotter - Ptmax
    #     plotter_rate = Plotter(x_val, rate_series, x_type='Ptmax', plot_type='Data Rate', smooth=False, combined_plot=False)
    #     plotter_energy = Plotter(x_val, energy_series, x_type='Ptmax', plot_type='Energy Efficiency', smooth=False, combined_plot=False)
        
    #     # # plotter - N
    #     # plotter_rate = Plotter(x_val, rate_series, x_type='N', plot_type='Data Rate', smooth=False, combined_plot=False)
    #     # plotter_energy = Plotter(x_val, energy_series, x_type='N', plot_type='Energy Efficiency', smooth=False, combined_plot=False)
        
    #     # # Plotter - PRmax
    #     # plotter_rate = Plotter(x_val, rate_series, x_type='PRmax', plot_type='Data Rate', smooth=True, combined_plot=False)
    #     # plotter_energy = Plotter(x_val, energy_series, x_type='PRmax', plot_type='Energy Efficiency', smooth=True, combined_plot=False)
        
    #     # # plotter - Pcn
    #     # plotter_energy = Plotter(x_val, energy_series, x_type='Pcn', plot_type='Energy Efficiency', smooth=False, combined_plot=False)
        
    #     # # plotter - a
    #     # plotter_energy = Plotter(x_val, energy_series, x_type='a', plot_type='Energy Efficiency', smooth=True, combined_plot=False)
        
    #     # # plotter - NEV
    #     # plotter_rate = Plotter(x_val, rate_series, x_type='NEV', plot_type='Data Rate', smooth=False, combined_plot=False)
    #     # plotter_energy = Plotter(x_val, energy_series, x_type='NEV', plot_type='Energy Efficiency', smooth=True, combined_plot=False)
        
        # # # plot_results - Ptmax
        # plotter_rate.plot_results(save_path='data/figures/TIF_figs', formats=['eps', 'png', 'pdf'], fig_name='figure_SSR_Vs_Ptmax') # save_path='data/figures/SSR_figs'
        # plotter_energy.plot_results(save_path='data/figures/TIF_figs', formats=['eps', 'png', 'pdf'], fig_name='figure_SEE_Vs_Ptmax') # save_path='data/figures/SEE_figs'
        
    #     # # plot_results - N
    #     # plotter_rate.plot_results(save_path=None, formats=['eps', 'png', 'pdf'], fig_name='figure_SSR_Vs_N') # save_path='data/figures/SSR_figs'
    #     # plotter_energy.plot_results(save_path=None, formats=['eps', 'png', 'pdf'], fig_name='figure_SEE_Vs_N_0dB') # 'data/figures/SEE_figs', None
        
    #     # # plot_results - PRmax
    #     # plotter_rate.plot_results(save_path='data/figures/SSR_figs', formats=['eps', 'png', 'pdf'], fig_name='figure_SSR_Vs_PRmax_0dB') # save_path='data/figures/SSR_figs'
    #     # plotter_energy.plot_results(save_path='data/figures/SEE_figs', formats=['eps', 'png', 'pdf'], fig_name='figure_SEE_Vs_PRmax_0dB') # save_path='data/figures/SEE_figs'
        
    #     # # plot_results - Pcn
    #     # plotter_energy.plot_results(save_path='data/figures/TIF_figs', formats=['eps', 'png', 'pdf'], fig_name='figure_SEE_Vs_Pcn_sCSI') # 'data/figures/SEE_figs'
        
    #     # # plot_results - a
    #     'data/figures/SEE_figs'
    #     # plotter_energy.plot_results(save_path=None, formats=['eps', 'png', 'pdf'], fig_name='figure_SEE_Vs_a') # 'data/figures/SEE_figs'
        
    #     # # plot_results - NEV
    #     # plotter_rate.plot_results(save_path=None, formats=['eps', 'png', 'pdf'], fig_name='figure_SSR_Vs_NEV') # save_path='data/figures/SSR_figs'
    #     # plotter_energy.plot_results(save_path='data/figures/TIF_figs', formats=['eps', 'png', 'pdf'], fig_name='figure_SEE_Vs_NEV') # 'data/figures/SEE_figs', None
    
    # Keep the figures open
    plt.ioff()
    plt.show()

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










# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.signal import savgol_filter
# from config import SystemConfig

# class Plotter:
#     def __init__(self, x_val, data_series, x_type='power', loc='upper left', plot_type='Data Rate', 
#                  smooth=False, window_length=5, polyorder=2):
#         self.x_val = x_val
#         self.data_series = data_series
#         self.x_type = x_type
#         self.loc = loc
#         self.plot_type = plot_type
#         self.smooth = smooth
#         self.window_length = window_length
#         self.polyorder = polyorder
    
#     def plot_results(self):
#         plt.figure(figsize=(10, 6))
#         self._set_labels()
        
#         for series in self.data_series:
#             y_data = self._apply_smoothing(series['data']) if self.smooth else series['data']
#             self._plot_data(series, y_data)
        
#         self._finalize_plot()
#         plt.show()

#     def _set_labels(self):
#         plt.xlabel(self._get_xlabel(), fontsize=15, fontweight='bold')
#         plt.ylabel(self._get_ylabel(), fontsize=15, fontweight='bold')

#     def _get_xlabel(self):
#         return "Maximum available Transmit Power (dBm)" if self.x_type == 'power' else "Maximum Gain (dB) of the RIS"  #"Number of RIS elements"

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

#     def _plot_data(self, series, y_data):
#         plot_methods = {
#             'Data Rate': plt.plot,
#             'Energy Efficiency': plt.plot,
#             'Bar': plt.bar,
#             'Scatter': plt.scatter
#         }
#         plot_method = plot_methods.get(self.plot_type, plt.plot)
        
#         if self.plot_type in ['Bar', 'Scatter']:
#             plot_method(self.x_val, y_data, label=series['label'], color=series['color'])
#         else:
#             plot_method(self.x_val, y_data, label=series['label'], color=series['color'], 
#                         marker=series['marker'], markersize=8, linewidth=series.get('line_width', 3))

#     def _finalize_plot(self):
#         plt.tick_params(axis='both', labelsize=15)
#         plt.xticks(fontweight='bold')
#         plt.yticks(fontweight='bold')
#         legend = plt.legend(fontsize=15, loc=self.loc)
#         legend.get_frame().set_facecolor('white')


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


# # Example Usage
# if __name__ == "__main__":
    
#     from utils import Utils
    
#     # Enable interactive mode for matplotlib
#     plt.ion()
    
#     # Initialize system configuration
#     config = SystemConfig()
    
#     # Load the results
    
#     # 1. Varying Ptmax
#     results_sr = np.load('data/outputs/output_results_algo1_sr_active_2s_100ris_10.0dB_5dBm_Ptmax.npz', allow_pickle=True)['arr_0'].item()  
#     results_ee = np.load('data/outputs/output_results_algo1_ee_active_2s_100ris_10.0dB_5dBm_Ptmax.npz', allow_pickle=True)['arr_0'].item()  
    
#     results_avg_sr = Utils.average_values_by_key_equal_length(results_sr, config.NUM_SAMPLES)
#     results_avg_ee = Utils.average_values_by_key_equal_length(results_ee, config.NUM_SAMPLES)
    
#     # Extract data for plotting
#     x_val = config.power_range_dbm #results['Ptmax'][0]
    
#     # results_ee = np.load('data/outputs/output_results_algo1_ee_active_40dBm_1s_N.npz', allow_pickle=True)['arr_0'].item() # config.OUTPUT_FILE
#     # results_sr = np.load('data/outputs/output_results_algo1_sr_active_40dBm_1s_N.npz', allow_pickle=True)['arr_0'].item()
#     # results_avg_sr = Utils.average_values_by_key_equal_length(results_sr, config.NUM_SAMPLES)
    
    
#     # 2. varying: a
#     # results_ee_1 = np.load('data/outputs/output_results_algo1_ee_active_2s_100ris_1e-05dBm_5dBm_a.npz', allow_pickle=True)['arr_0'].item()
#     # results_ee_2 = np.load('data/outputs/output_results_algo1_ee_active_2s_100ris_0.001dBm_5dBm_a.npz', allow_pickle=True)['arr_0'].item()
#     # results_ee_3 = np.load('data/outputs/output_results_algo1_ee_active_2s_100ris_0.1dBm_5dBm_a.npz', allow_pickle=True)['arr_0'].item()
    
#     # results_avg_ee_1 = Utils.average_values_by_key_equal_length(results_ee_1, config.NUM_SAMPLES)
    
#     # results_avg_ee_2 = Utils.average_values_by_key_equal_length(results_ee_2, config.NUM_SAMPLES)
    
#     # results_avg_ee_3 = Utils.average_values_by_key_equal_length(results_ee_3, config.NUM_SAMPLES)

#     # Extract data for plotting
#     # x_val = np.arange(0, 10.5, 0.5)
  
    
#     if config.quantization:
    
#         rate_series_Q_pcsi = [
#             {'data': results_avg_sr['ssr_sol_pcsi'], 'label': 'SSR Max. - pCSI', 'color': 'red', 'marker': 'o'},
#             {'data': results_avg_sr['ssr_sol_Q_pcsi'][(1,1)], 'label': 'SSR_Q Max. - pCSI - (1,1) bits', 'color': 'cyan', 'marker': 'x'},
#             {'data': results_avg_sr['ssr_sol_Q_pcsi'][(2,2)], 'label': 'SSR_Q Max. - pCSI - (2,2) bits', 'color': 'green', 'marker': 'x'},
#             {'data': results_avg_sr['ssr_sol_Q_pcsi'][(3,3)], 'label': 'SSR_Q Max. - pCSI - (3,3) bits', 'color': 'blue', 'marker': 'x'},
#             {'data': results_avg_sr['ssr_sol_Q_pcsi'][(4,4)], 'label': 'SSR_Q Max. - pCSI - (4,4) bits', 'color': 'magenta', 'marker': 'x'}
#         ]
        
#         rate_series_Q_scsi = [
#             {'data': results_avg_sr['ssr_sol_scsi'], 'label': 'SSR Max. - sCSI', 'color': 'red', 'marker': 'o'},
#             {'data': results_avg_sr['ssr_sol_Q_scsi'][(1,1)], 'label': 'SSR_Q Max. - sCSI - (1,1) bits', 'color': 'cyan', 'marker': 'x'},
#             {'data': results_avg_sr['ssr_sol_Q_scsi'][(2,2)], 'label': 'SSR_Q Max. - sCSI - (2,2) bits', 'color': 'green', 'marker': 'x'},
#             {'data': results_avg_sr['ssr_sol_Q_scsi'][(3,3)], 'label': 'SSR_Q Max. - sCSI - (3,3) bits', 'color': 'blue', 'marker': 'x'},
#             {'data': results_avg_sr['ssr_sol_Q_scsi'][(4,4)], 'label': 'SSR_Q Max. - sCSI - (4,4) bits', 'color': 'magenta', 'marker': 'x'}
#         ]
        
#         energy_series_Q_pcsi = [
#             {'data': results_avg_ee['see_sol_pcsi'], 'label': 'SEE Max. - pCSI', 'color': 'red', 'marker': 'o'},
#             {'data': results_avg_ee['see_sol_Q_pcsi'][(1,1)], 'label': 'SEE_Q Max. - pCSI - (1,1) bits', 'color': 'cyan', 'marker': 'x'},
#             {'data': results_avg_ee['see_sol_Q_pcsi'][(2,2)], 'label': 'SEE_Q Max. - pCSI - (2,2) bits', 'color': 'green', 'marker': 'x'},
#             {'data': results_avg_ee['see_sol_Q_pcsi'][(3,3)], 'label': 'SEE_Q Max. - pCSI - (3,3) bits', 'color': 'blue', 'marker': 'x'},
#             {'data': results_avg_ee['see_sol_Q_pcsi'][(4,4)], 'label': 'SEE_Q Max. - pCSI - (4,4) bits', 'color': 'magenta', 'marker': 'x'}
#         ]
        
#         energy_series_Q_scsi = [
#             {'data': results_avg_ee['see_sol_pcsi'], 'label': 'SEE Max. - sCSI', 'color': 'red', 'marker': 'o'},
#             {'data': results_avg_ee['see_sol_Q_pcsi'][(1,1)], 'label': 'SEE_Q Max. - sCSI - (1,1) bits', 'color': 'cyan', 'marker': 'x'},
#             {'data': results_avg_ee['see_sol_Q_pcsi'][(2,2)], 'label': 'SEE_Q Max. - p]sCSI - (2,2) bits', 'color': 'green', 'marker': 'x'},
#             {'data': results_avg_ee['see_sol_Q_pcsi'][(3,3)], 'label': 'SEE_Q Max. - sCSI - (3,3) bits', 'color': 'blue', 'marker': 'x'},
#             {'data': results_avg_ee['see_sol_Q_pcsi'][(4,4)], 'label': 'SEE_Q Max. - sCSI - (4,4) bits', 'color': 'magenta', 'marker': 'x'}
#         ]
        
#         # Plot results
#         plotter_rate_Q = Plotter(x_val, rate_series_Q, x_type='power', plot_type='Data Rate', smooth=False) # x_type='ris_elements'
#         plotter_energy_Q = Plotter(x_val, energy_series_Q, x_type='ris_gain', plot_type='Energy Efficiency', smooth=True) # x_type='power',x_type='ris_elements'
        
#         plotter_rate_Q.plot_results()
#         plotter_energy_Q.plot_results() 
           
#     rate_series = [
#         {'data': results_avg_sr['ssr_sol_pcsi'], 'label': 'SSR Max. - pCSI', 'color': 'blue', 'marker': 'o'},
#         {'data': results_avg_sr['ssr_sol_scsi'], 'label': 'SSR Max. - sCSI', 'color': 'red', 'marker': 'x'},
#          {'data': results_avg_ee['ssr_sol_pcsi'], 'label': 'SSR with SEE Max. - pCSI', 'color': 'green', 'marker': 'o'},
#         {'data': results_avg_ee['ssr_sol_scsi'], 'label': 'SSR with SEE Max. - sCSI', 'color': 'magenta', 'marker': 'x'},
#         {'data': results_avg_sr['ssr_uniform_pcsi'], 'label': 'SSR with random phases - pCSI', 'color': 'black', 'marker': 'o'},
#         {'data': results_avg_sr['ssr_uniform_scsi'], 'label': 'SSR with random phases - sCSI', 'color': 'cyan', 'marker': 'x'}
#     ]
    
#     energy_series = [
#         {'data': results_avg_ee['see_sol_pcsi'], 'label': 'SEE Max. - pCSI', 'color': 'blue', 'marker': 'o'},
#         {'data': results_avg_ee['see_sol_scsi'], 'label': 'SEE Max. - sCSI', 'color': 'red', 'marker': 'x'},
#         {'data': results_avg_sr['see_sol_pcsi'], 'label': 'SEE with SSR Max. - pCSI', 'color': 'green', 'marker': 'o'},
#         {'data': results_avg_sr['see_sol_scsi'], 'label': 'SEE with SSR. Max. - sCSI', 'color': 'magenta', 'marker': 'x'},
#         {'data': results_avg_ee['see_uniform_pcsi'], 'label': 'SEE with random phases - pCSI', 'color': 'black', 'marker': 'o'},
#         {'data': results_avg_ee['see_uniform_scsi'], 'label': 'SEE with random phases - sCSI', 'color': 'cyan', 'marker': 'x'}
#     ]

    
#     # Plot results
#     plotter_rate = Plotter(x_val, rate_series, x_type='power', plot_type='Data Rate', smooth=False) # x_type='ris_elements'
#     plotter_energy = Plotter(x_val, energy_series, x_type='ris_gain', plot_type='Energy Efficiency', smooth=False) # x_type='power', x_type='ris_elements'
    
#     # plotter_rate.plot_results()
#     plotter_energy.plot_results() 
    
#     # Keep the figures open
#     plt.ioff()
#     plt.show()

#     # # Example data for testing
#     # x_val = np.linspace(0, 10, 100)
#     # data_series = [
#     #     {'data': np.sin(x_val), 'label': 'Sine Wave', 'color': 'blue', 'marker': 'o'},
#     #     {'data': np.cos(x_val), 'label': 'Cosine Wave', 'color': 'red', 'marker': 'x'}
#     # ]
#     # plotter = Plotter(x_val, data_series, x_type='power', plot_type='Data Rate', smooth=True)
#     # plotter.plot_results()

#     # # Example data for 3D plotting
#     # RIS = [0, 0, 10]
#     # Tx = [[10, 0, 0], [0, 10, 0], [-10, 0, 0], [0, -10, 0]]
#     # Rx_B = [5, 5, 0]
#     # Rx_E = [-5, -5, 0]
#     # plot3d = Plot3DPositions(RIS, Tx, Rx_B, Rx_E)
#     # plot3d.plot()
