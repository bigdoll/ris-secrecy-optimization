import numpy as np

class SystemConfig:
    def __init__(self):
        # RIS and Optimization States
        self.ris_state = 'active'  # 'active' OR 'passive'
        self.cons_state = 'global'  # 'global' OR 'local'
        self.opt_state = 'ee'  # 'sr' OR 'ee'
        self.opt_bool = {'sr': 0, 'ee': 1}  # Optimization state

        # Quantization Settings
        self.quantization = False
        self.bits_phase = 3
        self.bits_amplitude = 3

        # Network Parameters
        self.K = 4  # Number of UEs
        self.Nt = 1  # Number of transmit antennas for each UE
        self.NR = [4, 1]  # Number of receive antennas at the BS (Bob) and Eve
        self.BW = 20e6  # System bandwidth in Hz
        self.BW0 = 1  # Normalized bandwidth
        self.mu = 1  # Amplifier inefficiency

        # RIS Amplification Factor
        self.a = 2 if self.ris_state == 'active' else 1

        # Power Consumption Settings (dBm)
        self.P0 = 40  # Total static power consumption
        self.Pcn_a = 5  # Active RIS element power consumption
        self.Pcn_p = 0  # Passive RIS element power consumption
        self.P0_RIS_a = 20  # Static power at active RIS
        self.P0_RIS_p = 10  # Static power at passive RIS

        # Noise and Interference Parameters
        self.k = 1.38e-23  # Boltzmann constant in J/K
        self.T = 290  # Room temperature in K
        self.NF = 5  # Noise figure in dB
        self.N0_dBm_Hz = self.calculate_n0_dbm_hz(self.k, self.T, self.NF)  # Noise power spectral density in dBm/Hz
        self.N0 = self.dbm_to_watts(self.N0_dBm_Hz)  # Noise power spectral density in W/Hz
        self.SNR_E_dB = 10  # Signal-to-noise ratio for Eve in dB
        self.SNR_E_linear = self.db_to_linear(self.SNR_E_dB)  # Convert SNR from dB to linear scale
        self.NV_E = self.calculate_channel_estimation_error_variance(self.N0, self.BW, self.SNR_E_linear)  # Noise variance for Eve's channel estimation error

        # Frequency and Wavelength
        self.f0 = 3.5e9  # Operating frequency in Hz
        self.c = 3e8  # Speed of light in m/s
        self.lambda_0 = self.c / self.f0  # Wavelength in meters
        self.d = self.lambda_0 / 2  # Half-wavelength RIS element spacing

        # Path-Loss Exponents and Rician Factors
        self.n = [4, 2, 4]  # Path-loss exponents: [n_h, n_B_g, n_E_g]
        self.R_K = [2, 4, 2]  # Rician factors: [K_h, K_B_g, K_E_g]

        # Coverage and Position Parameters
        self.d0 = 35  # Reference distance in meters
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

        # Dependent Parameters (Calculated)
        self.sigma_RIS_sq = self.calculate_noise_power_variance(self.N0, self.BW0)
        self.sigma_g_sq = self.NV_E

        # Simulation Parameters
        self.N_RANGE = range(10, 210, 10)
        self.NUM_SAMPLES = 10
        self.PTMAX_DBM = 40
        self.PTMAX = self.dbm_to_watts(self.PTMAX_DBM)
        self.FILENAME_CHANNEL = 'channel_samples_algo1_N_new.npy'
        self.FILENAME_RIS = "ris_coefficients_samples_algo1_N_new.npy"
        self.OUTPUT_FILE = f"./data/outputs/output_results_algo1_{self.opt_state}_{self.ris_state}_{self.PTMAX_DBM}dBm_{self.NUM_SAMPLES}s_N.npz"
        self.NAMES = [
            "sample_index", "n_index", "N", "Ptmax", "p_uniform", "gamma_random", "p_sol_pcsi", "p_sol_scsi",
            "gamma_sol_pcsi", "gamma_sol_scsi", "sr_uniform_Bob_pcsi", "sr_uniform_Bob_scsi", "sr_uniform_Eve_pcsi",
            "sr_uniform_Eve_scsi", "ssr_uniform_pcsi", "ssr_uniform_scsi", "gee_uniform_Bob_pcsi", "gee_uniform_Bob_scsi",
            "gee_uniform_Eve_pcsi", "gee_uniform_Eve_scsi", "see_uniform_pcsi", "see_uniform_scsi", "ssr_sol_pcsi",
            "ssr_sol_scsi", "see_sol_pcsi", "see_sol_scsi", "iteration_altopt_pcsi", "iteration_altopt_scsi",
            "iteration_p_pcsi", "iteration_p_scsi", "iteration_gamma_pcsi", "iteration_gamma_scsi",
            "time_complexity_altopt_pcsi", "time_complexity_altopt_scsi", "time_complexity_p_pcsi",
            "time_complexity_p_scsi", "time_complexity_gamma_pcsi", "time_complexity_gamma_scsi"
        ]

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

    def compute_path_loss_coefficient(self, d, n):
        """Compute path loss coefficient based on distance and path loss exponent."""
        PLo = (4 * np.pi * self.f0 * self.d0 / self.c) ** (-2)
        return np.sqrt(2 * PLo) / np.sqrt(1 + (d / self.d0) ** n)

    def compute_static_power_consumption(self, N):
        """Compute the total static power consumption."""
        P0_W = self.dbm_to_watts(self.P0)
        Pcn_W = self.dbm_to_watts(self.Pcn_a if self.ris_state == 'active' else self.Pcn_p)
        P0_RIS_W = self.dbm_to_watts(self.P0_RIS_a if self.ris_state == 'active' else self.P0_RIS_p)
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



# # Example of how to use the configuration
# config = SystemConfig(SNR_E_dB=10)

# # Access configuration parameters
# print(config.get_config_summary())

# # Example function call using config
# path_loss_coefficient = config.compute_path_loss_coefficient(100, config.n[0])
# print(f"Path Loss Coefficient: {path_loss_coefficient}")
