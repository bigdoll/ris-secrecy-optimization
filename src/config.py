import numpy as np
import re

class SystemConfig:
    def __init__(self):
        # RIS and Optimization States
        self.rf_state = 'RF-Gain' # 'RF-Gain' or 'RF-Power'
        self.ris_state = 'active'  # 'active' OR 'passive'
        self.cons_state = 'global'  # 'global' OR 'local'
        self.opt_state = 'ee'  # 'sr' OR 'ee'
        self.gamma_method = 'ls' # 'cvx1' 'cvx2' OR 'ls'
        self.opt_bool = {'sr': 0, 'ee': 1}  # Optimization state
        self.run_separately = True  # Run the simulation separately for each combination of varying parameters

        # Quantization Settings
        self.quantization = True #True
        self.bits_range = [(1,1), (2,2), (3,3), (4,4)] # [(bits_phase, bits_amplitude)]

        # Network Parameters
        self.K = 4  # Number of UEs
        self.Nt = 1  # Number of transmit antennas for each UE
        self.NR = [4, 1]  # Number of receive antennas at the BS (Bob) and Eve
        self.N  = 100 # 100 # Number of RIS antennas
        self.BW = 20e6  # System bandwidth in Hz
        self.BW0 = 1  # Normalized bandwidth
        self.mu = 1  # Amplifier inefficiency

        # RIS Amplification Factor : dB (e.g 3 dB, 5 dB or 10 dB)
        self.a = self.db_to_linear(5) if self.ris_state == 'active' else 1
        
        # RIS DC power control coefficient
        # self.q = 1 # [0-1]
        
        # Max RF power allocation at the RIS in dBm (0 or 30)
        self.PRmax = self.dbm_to_watts(0) if self.ris_state == 'active' else 0

        # Power Consumption Settings (dBm)
        self.P0 = 40  # Total static power consumption
        self.Pcn_a = 5  # Active RIS element power consumption in dBm
        self.Pcn_p = 0  # Passive RIS element power consumption in dBm
        self.Pcn = self.Pcn_a if self.ris_state == 'active' else self.Pcn_p
        self.P0_RIS_a = 20  # Static power at active RIS
        self.P0_RIS_p = 10  # Static power at passive RIS

        # Noise and Interference Parameters
        self.k = 1.38e-23  # Boltzmann constant in J/K
        self.T = 290  # Room temperature in K
        self.NF = 5  # Noise figure in dB
        self.N0_dBm_Hz = self.calculate_n0_dbm_hz(self.k, self.T, self.NF)  # Noise power spectral density in dBm/Hz
        self.N0 = self.dbm_to_watts(self.N0_dBm_Hz)  # Noise power spectral density in W/Hz
        self.NEEV_dB = 0 # 0  # Normalized Estimation Error Variance (NEEV) for Eve in dB
        self.NEEV = self.db_to_linear(self.NEEV_dB)  # Convert SNR from dB to linear scale
        self.vic_percent_eve = 1 # 50 / 100
        # self.NV_E = self.calculate_channel_estimation_error_variance(self.N0, self.BW, self.SNR_E)  # Noise variance for Eve's channel estimation error

        # Frequency and Wavelength
        self.f0 = 3.5e9  # Operating frequency in Hz
        self.c = 3e8  # Speed of light in m/s
        self.lambda_0 = self.c / self.f0  # Wavelength in meters
        self.d = self.lambda_0 / 2  # Half-wavelength RIS element spacing

        # Path-Loss Exponents and Rician Factors
        self.n = [4, 2, 4]  # Path-loss exponents: [n_h, n_B_g, n_E_g]
        self.R_K = [2, 4, 2]  # Rician factors: [K_h, K_B_g, K_E_g]

        # Coverage and Position Parameters
        self.d0 = 35 # 35  # Reference distance in meters
        self.r_cell = 50  # Radius of the cell in meters
        self.h_BS = 10  # Height of the BS in meters
        self.h_RIS = 10  # Height of the RIS in meters
        self.hmin_UE = 1.5  # Minimum height for UEs in meters
        self.hmax_UE = 2.5  # Maximum height for UEs in meters

        # Power Allocation Parameters
        self.power_min_dbm = -20  # Minimum power in dBm
        self.power_max_dbm = 50  # Maximum power in dBm
        self.power_step_dbm = 2  # Power step in dBm
        self.power_range_dbm, self.power_range_watts = self.set_power_range(
            self.power_min_dbm, self.power_max_dbm, self.power_step_dbm)
        self.Ptmax = self.dbm_to_watts(10) # Combined Maximum Transmit Power of the UEs dBm : 30 dBm, 0dBm

        # Dependent Parameters (Calculated)
        self.sigma_sq = self.calculate_noise_power_variance(self.N0, self.BW)
        self.sigma_RIS_sq = self.calculate_noise_power_variance(self.N0, self.BW)
        # self.sigma_g_sq = self.NV_E
        
        

        # Simulation Parameters
        self.simulation_type = None
        # self.N_RANGE = range(100, 110, 10) # range(10, 210, 10)
        self.NUM_SAMPLES = 2 #10
        self.channel_model = "rician"
        self.PTMAX_DBM = 50
        self.PTMAX = self.dbm_to_watts(self.PTMAX_DBM)
        self.PTMIN_DBM = -20 #-20
        self.PTMIN = self.dbm_to_watts(self.PTMIN_DBM)
        
        self.fixed_params = {'Ptmax': self.Ptmax, 'N': self.N, 'PRmax': self.PRmax, 'a': self.a, 'Pcn_p': self.Pcn_p, 'Pcn_a': self.Pcn_a}
        
        # self.varying_params = self.fixed_params
        # {'Ptmax': self.PTMAX_DBM, 'N': self.N, 'a': self.a, 'Pcn_p': self.Pcn_p, 'Pcn_a': self.Pcn_a}
        
        # # channels - N
        # self.FILENAME_CHANNEL = f'channel_samples_algo1_{self.NUM_SAMPLES}s_{self.r_cell}r_{self.vic_percent_eve * 100}%vic_{self.NEEV_dB}dBvar_N.npy'
        # self.FILENAME_RIS = f"ris_coefficients_samples_algo1_{self.NUM_SAMPLES}s_{self.r_cell}r_{self.vic_percent_eve * 100}%vic_{self.NEEV_dB}dBvar_N.npy"
        
        # # channels - NEEV
        # self.FILENAME_CHANNEL = f'channel_samples_algo1_{self.channel_model}_{self.NUM_SAMPLES}s_{self.r_cell}r_{self.vic_percent_eve * 100}%vic_{self.N}ris_NEEV_test.npy'
        # self.FILENAME_RIS = f"ris_coefficients_samples_algo1_{self.NUM_SAMPLES}s_{self.N}ris_NEEV_test.npy"
        
        # # channels -  Ptmax, a, Pcn, PRmax
        self.FILENAME_CHANNEL = f'channel_samples_algo1_{self.NUM_SAMPLES}s_{self.r_cell}r_{self.N}ris_{self.NEEV_dB}dBvar.npy'
        self.FILENAME_RIS = f"ris_coefficients_samples_algo1_{self.NUM_SAMPLES}s_{self.r_cell}r_{self.N}ris_{self.NEEV_dB}dBvar.npy"
        
        # self.OUTPUT_FILE = f"./data/outputs/output_results_algo1_{self.opt_state}_{self.ris_state}_{self.NUM_SAMPLES}s_{self.PTMAX_DBM}dBm_N.npz"
        
        self.set_output_file()
        
        self.NAMES = [
            "sample_index", "params_combo", "p_uniform", "gamma_random", "p_sol_pcsi", "p_sol_scsi",
            "gamma_sol_pcsi", "gamma_sol_Q_pcsi", "gamma_sol_scsi", "gamma_sol_Q_scsi", "sr_uniform_Bob_pcsi", "sr_uniform_Bob_scsi", "sr_uniform_Eve_pcsi",
            "sr_uniform_Eve_scsi", "ssr_uniform_pcsi", "ssr_uniform_scsi", "gee_uniform_Bob_pcsi", "gee_uniform_Bob_scsi",
            "gee_uniform_Eve_pcsi", "gee_uniform_Eve_scsi", "see_uniform_pcsi", "see_uniform_scsi", "sr_sol_pcsi", "sr_sol_Q_pcsi", "ssr_sol_pcsi", "ssr_sol_Q_pcsi",
            "sr_sol_scsi", "sr_sol_Q_scsi", "ssr_sol_scsi", "ssr_sol_Q_scsi", "gee_sol_pcsi", "gee_sol_Q_pcsi", "see_sol_pcsi", "see_sol_Q_pcsi", "gee_sol_scsi", "gee_sol_Q_scsi",  "see_sol_scsi", "see_sol_Q_scsi", "iteration_altopt_pcsi", "iteration_altopt_scsi",
            "iteration_p_pcsi", "iteration_p_scsi", "iteration_gamma_pcsi", "iteration_gamma_scsi",
            "time_complexity_altopt_pcsi", "time_complexity_altopt_scsi", "time_complexity_p_pcsi",
            "time_complexity_p_scsi", "time_complexity_gamma_pcsi", "time_complexity_gamma_scsi"
        ] # "n_index", "N", "p_index",  "Ptmax",

    @staticmethod
    def db_to_linear(db_value):
        """Convert dB to linear scale."""
        return 10 ** (db_value / 10)

    @staticmethod
    def dbm_to_watts(power_dbm):
        """Convert power from dBm to Watts."""
        return 10 ** ((power_dbm - 30) / 10)

    @staticmethod
    def calculate_noise_power_variance(N0, BW):
        """Calculate noise power variance."""
        return N0 * BW

    @staticmethod
    def calculate_n0_dbm_hz(k, T, NF):
        """Calculate the noise power spectral density in dBm/Hz."""
        P_thermal = k * T
        P_thermal_dBm_Hz = 10 * np.log10(P_thermal) + 30  # Convert from W/Hz to dBm/Hz
        return P_thermal_dBm_Hz + NF

    @staticmethod
    def calculate_channel_estimation_error_variance(N0, BW, SNR):
        """Calculate the noise variance for channel estimation error."""
        return N0 * BW / SNR

    @staticmethod
    def set_power_range(power_min_dbm, power_max_dbm, power_step_dbm):
        """Set the power range in dBm and convert it to Watts."""
        power_range_dbm = np.arange(power_min_dbm, power_max_dbm + power_step_dbm, power_step_dbm)
        power_range_watts = [10 ** ((p - 30) / 10) for p in power_range_dbm]  # Convert dBm to Watts
        return power_range_dbm, power_range_watts

    def set_output_file(self):
        if self.simulation_type == 1: #'Ptmax'
            self.OUTPUT_FILE = f"./data/output/output_results_algo1_{self.opt_state}_{self.ris_state}_{self.NUM_SAMPLES}s_{self.N}ris_{10*np.log10(self.a)}dB_{self.Pcn}dBm_{self.NEEV_dB}dBvar_Ptmax.npz"
        elif self.simulation_type == 2: # 'N'
            self.OUTPUT_FILE = f"./data/outputs/output_results_algo1_{self.opt_state}_{self.ris_state}_{self.NUM_SAMPLES}s_{10*np.log10(self.Ptmax) + 30}dBm_{10*np.log10(self.a)}dB_{self.Pcn}dBm_{self.NEEV_dB}dBvar_N.npz"
        elif self.simulation_type == 3: # 'PRmax'
            self.OUTPUT_FILE = f"./data/outputs/output_results_algo1_{self.opt_state}_{self.ris_state}_{self.NUM_SAMPLES}s_{self.N}ris_{10*np.log10(self.Ptmax) + 30}dBm_{self.Pcn}dBm_{self.NEEV_dB}dBvar_PRmax.npz"
        elif self.simulation_type == 4: # 'a'
            self.OUTPUT_FILE = f"./data/output_test/output_results_algo1_{self.opt_state}_{self.ris_state}_{self.NUM_SAMPLES}s_{self.N}ris_{10*np.log10(self.Ptmax) + 30}dBm_{self.Pcn}dBm_{self.NEEV_dB}dBvar_a.npz"
        elif self.simulation_type == 5: # 'Pcn'
            self.OUTPUT_FILE = f"./data/output_test/output_results_algo1_{self.opt_state}_{self.ris_state}_{self.NUM_SAMPLES}s_{self.N}ris_{10*np.log10(self.a)}dB_{10*np.log10(self.Ptmax) + 30}dBm_{self.NEEV_dB}dBvar_Pcn.npz"
        elif self.simulation_type == 6: # 'NEEV'
            self.OUTPUT_FILE = f"./data/output_test/output_results_algo1_{self.opt_state}_{self.ris_state}_{self.NUM_SAMPLES}s_{self.N}ris_{10*np.log10(self.Ptmax) + 30}dBm_{self.Pcn}dBm_{10*np.log10(self.a)}_NEEV.npz"
        else:
            self.OUTPUT_FILE = f"./data/outputs/output_results_algo1_{self.opt_state}_{self.ris_state}_{self.NUM_SAMPLES}s_default.npz"
    

    def update_output_file(self, simulation_type):
        self.simulation_type = simulation_type
        self.set_output_file()
    
    def update_custom_params(self, N=None, Ptmax=None, PRmax=None, a=None, Pcn_p=None, Pcn_a=None):
        if N is not None:
            self.N = N
        if Ptmax is not None:
            self.Ptmax = self.dbm_to_watts(Ptmax)
        if a is not None:
            self.a = self.db_to_linear(a)
        if PRmax is not None:
            self.PRmax = self.dbm_to_watts(PRmax)       
        if Pcn_p is not None:
            self.Pcn_p = Pcn_p
        if Pcn_a is not None:
            self.Pcn_a = Pcn_a
        # self.set_output_file()
    
    @staticmethod
    def compute_channel_estimation_error_variance_eve(
        pilot_powers: np.ndarray,  # shape (K,)
        H: np.ndarray,             # shape (N, K)
        sigma_E_sq: float,         # Eve’s AWGN variance
        sigma_gE_sq: float,        # prior var of g_E (per‐complex)
        sigma_RIS_sq: float        # per‐element RIS thermal noise
    ) -> float:
        """
        Compute scalar LMMSE error variance for estimating g_E via K pilots
        sent through the RIS.  We approximate the total noise at Eve as the
        sum of:
        - AWGN at Eve:         sigma_E_sq
        - aggregated RIS noise: sigma_RIS_sq * ||H||_F^2

        Then the LMMSE MSE for the N×1 vector g_E is
        Cov_e = ( H P H^H / sigma_v_sq + I/ sigma_gE_sq )^{-1},
        and we take its average variance = trace(Cov_e)/N.  Finally, we
        return that scalar.
        """
        N, K = H.shape

        # 1) “Pilot‐gain” matrix D^H D = H diag(p) H^H, but we only need its trace:
        pilot_gain = np.sum([
            pilot_powers[k] * np.linalg.norm(H[:, k])**2
            for k in range(K)
        ])

        # 2) Total noise variance at Eve (RIS noise folded in):
        noise_ris = sigma_RIS_sq * np.linalg.norm(H, ord='fro')**2
        sigma_v_sq = sigma_E_sq + noise_ris

        # 3) Scalar MSE via LMMSE for vector g_E:
        #    σ_e² = noise_total / ( pilot_gain + noise_total / sigma_gE_sq )
        sigma_e_sq = sigma_v_sq / (pilot_gain + sigma_v_sq / sigma_gE_sq)

        return sigma_e_sq
    
    # def compute_channel_estimation_error_variance_eve(pilot_powers, gE, H, sigma_E_sq, sigma_gE_sq, sigma_RIS_sq):
    #     """
    #     Compute the variance of the channel estimation error of the Eve channel.

    #     Parameters:
    #     - pilot_powers (array): The pilot powers of the UEs.
    #     - gE (array): The actual (truth) channel of RIS-Eve.
    #     - H (array): The channel matrix from RIS to UEs.
    #     - sigma_E_sq (float): The noise variance of the Eve channel.
    #     - sigma_gE_sq (float): The variance of the actual (truth) channel of RIS-Eve.
    #     - sigma_RIS_sq (float): The noise variance of the RIS.

    #     Returns:
    #     - float: The variance of the channel estimation error of the Eve channel.
    #     """
    #     K = pilot_powers.shape[0]
    #     P_tot_p = np.sum([pilot_powers[k] * np.linalg.norm(gE @ H[:, k])**2 for k in range(K)]) + sigma_RIS_sq * np.linalg.norm(gE)**2
    #     sigma_e_sq = sigma_E_sq / (P_tot_p + sigma_E_sq / sigma_gE_sq)
    #     return sigma_e_sq


    def compute_path_loss_coefficient(self, d, n):
        """Compute path loss coefficient based on distance and path loss exponent."""
        PLo = (4 * np.pi * self.f0 * self.d0 / self.c) ** (-2)
        return np.sqrt(2 * PLo) / np.sqrt(1 + (d / self.d0) ** n)

    def compute_static_power_consumption(self, N, state = None):
        """Compute the total static power consumption."""
        P0_W = self.dbm_to_watts(self.P0)
        Pcn_W = self.dbm_to_watts(self.Pcn_a if state == 'active' else self.Pcn_p)
        P0_RIS_W = self.dbm_to_watts(self.P0_RIS_a if state == 'active' else self.P0_RIS_p)
        return P0_W + N * Pcn_W + P0_RIS_W

    def set_ris_state(self, state):
        """Set the RIS state and update related parameters."""
        if state not in ['active', 'passive']:
            raise ValueError("Invalid RIS state. Choose 'active' or 'passive'.")
        self.ris_state = state
        self.a = 2 if self.ris_state == 'active' else 1

    def set_opt_state(self, state):
        """Set the optimization state."""
        if state not in ['sr', 'ee']:
            raise ValueError("Invalid optimization state. Choose 'sr' or 'ee'.")
        self.opt_state = state

    def get_config_summary(self):
        """Get a summary of the configuration parameters."""
        summary = (
            f"RIS State: {self.ris_state}\n"
            f"Optimization State: {self.opt_state}\n"
            f"Number of UEs: {self.K}\n"
            f"Operating Frequency: {self.f0} Hz\n"
            f"Noise Power Spectral Density: {self.N0_dBm_Hz} dBm/Hz\n"
            f"Noise Figure: {self.NF} dB\n"
            f"Power Step: {self.power_step_dbm} dBm\n"
            f"Total Static Power Consumption: {self.compute_static_power_consumption(100):.2f} W\n"
        )
        return summary
    
    @staticmethod
    def parse_param_string(param_string):
        # Regular expression to match key-value pairs without units
        pattern = re.compile(r'(\w+):\s*([\d\.]+)')
        
        # Find all matches in the input string
        matches = pattern.findall(param_string)
        
        # Construct the dictionary from matches
        param_dict = {key: float(value) if '.' in value else int(value) for key, value in matches}
        
        return param_dict



# # Example of how to use the configuration
# config = SystemConfig(SNR_E_dB=10)

# # Access configuration parameters
# print(config.get_config_summary())

# # Example function call using config
# path_loss_coefficient = config.compute_path_loss_coefficient(100, config.n[0])
# print(f"Path Loss Coefficient: {path_loss_coefficient}")
