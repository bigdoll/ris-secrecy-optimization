import os
import numpy as np
from scipy.linalg import sqrtm
from visualization import Plot3DPositions
from collections import defaultdict
from typing import Tuple, Dict
from typing import List, Dict, Any

class Utils:
    
    @staticmethod
    def check_file_exists(file_path):
        """
        Check if a file exists at the given path, creating it if it does not exist.

        Parameters:
        - file_path (str): Path of the file to check.

        Returns:
        - bool: True if the file exists, False otherwise.
        """
        exists = os.path.isfile(file_path)
        if exists:
            print("File already exists.")
            return True
        else:
            print("File not found. Creating a new file.")
            with open(file_path, 'w') as file:
                pass
            return False

    @staticmethod
    def get_file_path(file_name):
        """
        Construct the file path for the given file name in the ./data/samples directory.

        Parameters:
        - file_name (str): Name of the file.

        Returns:
        - str: Full file path.
        """
        return os.path.join("./data/samples", file_name)

    # @staticmethod
    def check_channel_or_ris_file_exists(generate_samples_func):
        """
        Decorator to check if a channel or RIS file exists before generating samples.

        Parameters:
        - generate_samples_func (callable): The function to generate samples.

        Returns:
        - callable: Wrapper function.
        """
        def wrapper_generate_samples(*args, **kwargs):
            filename = args[1] if args else kwargs.get('filename')
            if not filename:
                raise ValueError("Filename is missing.")
            file_path = Utils.get_file_path(filename)
            if os.path.exists(file_path):
                samples = np.load(file_path, allow_pickle=True)
                print("Samples loaded from file.")
            else:
                samples = generate_samples_func(*args, **kwargs)
                np.save(file_path, samples)
                print("Samples saved to file.")
            return samples
        return wrapper_generate_samples

    # #@staticmethod
    # @check_channel_or_ris_file_exists
    # def monte_carlo_simulation(num_samples, filename, *args, **kwargs):
    #     """
    #     Perform Monte Carlo simulation to generate channel samples.

    #     Parameters:
    #     - num_samples (int): Number of samples to generate.
    #     - filename (str): Name of the file to save the samples.

    #     Returns:
    #     - np.ndarray: Generated channel samples.
    #     """
    #     channel_samples = []
    #     for _ in range(num_samples):
    #         channels = Utils.generate_channels(*args, **kwargs)
    #         channel_samples.append(channels)
    #         # [Utils.generate_channels_algo1(*args, **kwargs) for _ in range(num_samples)]
    #     return Utils.create_object_array_from_tuples(channel_samples)
    
    @staticmethod
    def create_object_array_from_tuples(channel_samples):
        """
        Create an object array from a list of tuples.

        Parameters:
        - channel_samples (list): List of channel samples.

        Returns:
        - np.ndarray: Object array of channel samples.
        """
        samples_array = np.empty(len(channel_samples), dtype=object)
        for i, sample in enumerate(channel_samples):
            samples_array[i] = sample
        return samples_array
    
    @staticmethod
    @check_channel_or_ris_file_exists
    def monte_carlo_simulation(num_samples, filename, N_range, *args, **kwargs):
        """
        Perform Monte Carlo simulation to generate both channel samples and RIS coefficients.

        Parameters:
        - num_samples (int): Number of samples to generate.
        - filename (str): Name of the file to save the samples.
        - N_range (list): Range of RIS elements.

        Returns:
        - tuple: Generated channel samples and RIS coefficients.
        """
        channel_samples = []

        for _ in range(num_samples):
            channels = Utils.generate_channels(N_range, *args, **kwargs)
            channel_samples.append(channels)

        return Utils.create_object_array_from_tuples(channel_samples)
    
    @staticmethod
    @check_channel_or_ris_file_exists
    def generate_ris_coefficients(num_samples, filename, N_range, NEV_range, channel_samples_range, *args, **kwargs):
        """
        Generate random RIS reflection coefficients and save them in a file.

        Parameters:
        - num_samples (int): Number of samples to generate.
        - filename (str): Name of the file to save the samples.
        - Nris (list): Range of RIS elements.
        - channel_samples_range (list): Range of channel samples.

        Returns:
        - np.ndarray: Generated RIS coefficients.
        """
        #[]
        ris_samples = [] 
        for i in range(num_samples):
            ris_coefficients_samples = {}
            channel_samples = channel_samples_range[i]

            for N in N_range:
                ris_coefficients_nev = {}
                for nev in NEV_range:
                    
                    while True:
                        ris_coefficients = np.exp(1j * 2 * np.pi * np.random.rand(N, 1))
                        if kwargs['state'] == 'passive':
                            ssr_ris = np.random.rand()  # Placeholder for actual SSR computation
                            if ssr_ris > 0:
                                ris_coefficients_dict[N] = ris_coefficients
                                break
                        else:
                            sr_ris_Bob = Utils.SR_active_algo1(channel_samples[N]['G_B'], channel_samples[N]['H'], ris_coefficients, *args, channel_samples[N]['sigma_e_sq'][nev], scsi_bool=0, orig_bool=True, Rx="Bob")
                            sr_ris_Eve = Utils.SR_active_algo1(channel_samples[N]['gE_hat'] + channel_samples[N]['gE_error'][nev], channel_samples[N]['H'], ris_coefficients, *args, channel_samples[N]['sigma_e_sq'][nev], scsi_bool=1, orig_bool=False, Rx="Eve")
                            ssr_ris = sr_ris_Bob - sr_ris_Eve
                            if ssr_ris > 0:
                                ris_coefficients_dict = {
                                    'gamma' : ris_coefficients
                                    }
                                break
                    ris_coefficients_nev[nev] = ris_coefficients_dict
                ris_coefficients_samples[N] = ris_coefficients_nev # ris_coefficients_dict
                
                ris_samples.append(ris_coefficients_samples)
            #ris_coefficients_dict.append(ris_coefficients_dict)

        return Utils.create_object_array_from_tuples(ris_samples)
                        
    # @staticmethod
    # @check_channel_or_ris_file_exists
    # def generate_ris_coefficients(num_samples, filename, Nris, channel_samples_range, *args, **kwargs):
    #     """
    #     Generate random RIS reflection coefficients and save them in a file.

    #     Parameters:
    #     - num_samples (int): Number of samples to generate.
    #     - filename (str): Name of the file to save the samples.
    #     - N_range (list): Range of RIS elements.
    #     - channel_samples_range (list): Range of channel samples.

    #     Returns:
    #     - np.ndarray: Generated RIS coefficients.
    #     """
    #     ris_coefficients_samples = []

    #     for i in range(num_samples):
    #         ris_coefficients_lst = []
    #         G_B_lst, gE_hat_lst, gE_error_lst, H_lst, sigma_e_sq = channel_samples_range[i]
            

    #         for n_index, N in enumerate(Nris):
    #             while True:
    #                 ris_coefficients = np.exp(1j * 2 * np.pi * np.random.rand(N, 1))
    #                 if kwargs['state'] == 'passive':
    #                     ssr_ris = np.random.rand()  # Placeholder for actual SSR computation
    #                     if ssr_ris > 0:
    #                         ris_coefficients_samples.append(ris_coefficients)
    #                         break
    #                 else:
    #                     sr_ris_Bob = Utils.SR_active_algo1(G_B_lst[n_index], H_lst[n_index], ris_coefficients, *args, sigma_e_sq, scsi_bool=0, orig_bool=True, Rx="Bob")
    #                     sr_ris_Eve = Utils.SR_active_algo1(gE_hat_lst[n_index], H_lst[n_index], ris_coefficients, *args, sigma_e_sq, scsi_bool=1, orig_bool=False, Rx="Eve")
    #                     ssr_ris = sr_ris_Bob - sr_ris_Eve
    #                     if ssr_ris > 0:
    #                         ris_coefficients_lst.append(ris_coefficients)
    #                         break
    #         ris_coefficients_samples.append(ris_coefficients_lst)

    #     return Utils.create_object_array_from_tuples(ris_coefficients_samples)

    @staticmethod
    def generate_channels(N_range, r_cell, h_BS, h_RIS, hmin_UE, hmax_UE, d0, K, NR, f0, lambda_0, d, c, n, R_K, NEV_range, vic_percent_eve, channel_model="rician", shadowing=False):
        """
        Generate channel samples with optional shadowing effects.

        Parameters:
        - Various simulation parameters.
        - channel_model (str): Channel model to use ("rician", "rayleigh", "nakagami").
        - shadowing (bool): Whether to include shadowing effects.

        Returns:
        - dict: Generated channel samples for each N.
        """
        channel_samples = {}
        K_h, K_B_g, K_E_g = R_K
        NR_B, NR_E = NR

        # Calculate the near-field distance for all N values in the range
        N_x_y_pairs_RIS = [Utils.compute_N_x_y(N) for N in N_range]
        A_RIS_values = [Utils.calculate_ris_surface_size(N_x, N_y, wavelength=lambda_0, spacing=d) for N_x, N_y in N_x_y_pairs_RIS]
        Rn_RIS = [Utils.calculate_near_field_distance_from_area(A_RIS, wavelength=lambda_0) for A_RIS in A_RIS_values]
        
        N_x_y_pairs_BS = [Utils.compute_N_x_y(N) for N in [NR_B]]
        A_BS_values = [Utils.calculate_ris_surface_size(N_x, N_y, wavelength=lambda_0, spacing=d) for N_x, N_y in N_x_y_pairs_BS]
        Rn_B = [Utils.calculate_near_field_distance_from_area(A_BS, wavelength=lambda_0) for A_BS in A_BS_values]

        # Find the maximum near-field distance
        max_Rn_RIS = max(Rn_RIS)
        max_Rn_B = max(Rn_B)

        # Generate positions using the maximum near-field distance
        RIS_pos, Rx_B, Rx_E, Tx = Utils.generate_positions(K, r_cell, h_BS, h_RIS, hmin_UE, hmax_UE, max_Rn_B, max_Rn_RIS, vic_percent_eve)
        
        plot3d = Plot3DPositions(RIS_pos, Tx, Rx_B, Rx_E)
        plot3d.plot()

        dtx_ris = np.array([Utils.compute_path_length(RIS_pos, tx) for tx in Tx])
        drx_ris_B = Utils.compute_path_length(Rx_B, RIS_pos)
        drx_ris_E = Utils.compute_path_length(Rx_E, RIS_pos)

        for i, N in enumerate(N_range):
            sigma_e_sq = {}
            channel_gE_error = {}
            if shadowing:
                alpha_h = np.array([Utils.compute_path_loss_coefficient(d, n[0], c, f0, Rn_RIS[i]) * Utils.apply_shadowing(d) for d in dtx_ris])
                alpha_B_g = Utils.compute_path_loss_coefficient(drx_ris_B, n[1], c, f0, Rn_RIS[i]) * Utils.apply_shadowing(drx_ris_B)
                alpha_E_g = Utils.compute_path_loss_coefficient(drx_ris_E, n[2], c, f0, Rn_RIS[i]) * Utils.apply_shadowing(drx_ris_E)
            else:
                alpha_h = Utils.compute_path_loss_coefficient(dtx_ris, n[0], c, f0, Rn_RIS[i])
                alpha_B_g = Utils.compute_path_loss_coefficient(drx_ris_B, n[1], c, f0, Rn_RIS[i])
                alpha_E_g = Utils.compute_path_loss_coefficient(drx_ris_E, n[2], c, f0, Rn_RIS[i])
                
            sigma_gE_sq = alpha_E_g**2 * (1 /  (K_E_g + 1)) # (1 / (2 * (K_E_g + 1)))
            
            for _, nev_dB in enumerate(NEV_range):
                nev_linear = 10 ** (nev_dB / 10)
                sigma_e_sq[nev_dB] = sigma_gE_sq * nev_linear
                channel_gE_error[nev_dB] = Utils.generate_cscg_channel(N, sigma_e_sq[nev_dB])
            
            if channel_model == "rician":
                channel_H = np.zeros((N, K), dtype=np.complex128)
                for k in range(K):
                    channel_H[:, k] = alpha_h[k] * Utils.generate_rician_channel(N, K_h).flatten()
                channel_G_B = alpha_B_g * Utils.generate_rician_channel(NR_B * N, K_B_g).reshape(NR_B, N)
                channel_gE = alpha_E_g * Utils.generate_rician_channel(N * NR_E, K_E_g).reshape(N, NR_E)

            elif channel_model == "rayleigh":
                channel_H = np.array([alpha_h[k] * Utils.generate_rayleigh_channel(N) for k in range(K)]).T
                channel_G_B = alpha_B_g * Utils.generate_rayleigh_channel(NR_B * N).reshape(NR_B, N)
                channel_gE = alpha_E_g * Utils.generate_rayleigh_channel(N * NR_E).reshape(N, NR_E)

            elif channel_model == "nakagami":
                m = 1  # Nakagami shape parameter, can be adjusted
                channel_H = np.array([alpha_h[k] * Utils.generate_nakagami_channel(N, m) for k in range(K)]).T
                channel_G_B = alpha_B_g * Utils.generate_nakagami_channel(NR_B * N, m).reshape(NR_B, N)
                channel_gE = alpha_E_g * Utils.generate_nakagami_channel(N * NR_E, m).reshape(N, NR_E)

            else:
                raise ValueError("Unsupported channel model. Choose from 'rician', 'rayleigh', 'nakagami'.")


            sample_channels = {
                'G_B': channel_G_B,
                'gE_hat': channel_gE, # 'gE': channel_gE,
                'gE_error': channel_gE_error,
                'H': channel_H,
                'sigma_e_sq': sigma_e_sq
            }

            channel_samples[N] = sample_channels

        return channel_samples
    
    # @staticmethod
    # def generate_channels_algo1(r_cell, h_BS, h_RIS, hmin_UE, hmax_UE, d0, N_range, K, NR, f0, lambda_0, d, c, n, R_K, NEEV, vic_percent_eve, channel_model="rician", shadowing=False):
    #     """
    #     Generate channel samples with optional shadowing effects.

    #     Parameters:
    #     - Various simulation parameters.
    #     - channel_model (str): Channel model to use ("rician", "rayleigh", "nakagami").
    #     - shadowing (bool): Whether to include shadowing effects.

    #     Returns:
    #     - tuple: Generated channel samples.
    #     """
    #     channel_G_B_lst = []
    #     channel_gE_lst = []
    #     channel_gE_error_lst = []
    #     channel_H_lst = []
    #     # sigma_e_sq_lst = []

    #     # N_max = np.max(N_range)
    #     # N_x = N_y = int(np.sqrt(N_max))
    #     # A_RIS = Utils.calculate_ris_surface_size(N_x, N_y, wavelength=lambda_0, spacing=d)
    #     # Rn = Utils.calculate_near_field_distance_from_area(A_RIS, wavelength=lambda_0)
    #     # RIS_pos, Rx_B, Rx_E, Tx = Utils.generate_positions(K, r_cell, h_BS, h_RIS, hmin_UE, hmax_UE, Rn, vic_percent_eve)
        
    #     # Calculate the near-field distance for all N values in the range
    #     N_x_y_pairs = [Utils.compute_N_x_y(N) for N in N_range]
    #     A_RIS_values = [Utils.calculate_ris_surface_size(N_x, N_y, wavelength=lambda_0, spacing=d) for N_x, N_y in N_x_y_pairs]
    #     Rn = [Utils.calculate_near_field_distance_from_area(A_RIS, wavelength=lambda_0) for A_RIS in A_RIS_values]

    #     # Find the maximum near-field distance
    #     max_Rn = max(Rn)

    #     # Generate positions using the maximum near-field distance
    #     RIS_pos, Rx_B, Rx_E, Tx = Utils.generate_positions(K, r_cell, h_BS, h_RIS, hmin_UE, hmax_UE, max_Rn, vic_percent_eve)
        
    #     plot3d = Plot3DPositions(RIS_pos, Tx, Rx_B, Rx_E)
    #     plot3d.plot()

    #     dtx_ris = np.array([Utils.compute_path_length(RIS_pos, tx) for tx in Tx])
    #     drx_ris_B = Utils.compute_path_length(Rx_B, RIS_pos)
    #     drx_ris_E = Utils.compute_path_length(Rx_E, RIS_pos)

    #     K_h, K_B_g, K_E_g = R_K
    #     NR_B, NR_E = NR

    #     for i, N in enumerate(N_range):
            
    #         if shadowing:
    #             alpha_h = np.array([Utils.compute_path_loss_coefficient(d, n[0], c, f0, Rn[i]) * Utils.apply_shadowing(d) for d in dtx_ris]) #d0
    #             alpha_B_g = Utils.compute_path_loss_coefficient(drx_ris_B, n[1], c, f0, Rn[i]) * Utils.apply_shadowing(drx_ris_B) #d0
    #             alpha_E_g = Utils.compute_path_loss_coefficient(drx_ris_E, n[2], c, f0, Rn[i]) * Utils.apply_shadowing(drx_ris_E) # d0
    #         else:
    #             alpha_h = Utils.compute_path_loss_coefficient(dtx_ris, n[0], c, f0, Rn[i]) #d0
    #             alpha_B_g = Utils.compute_path_loss_coefficient(drx_ris_B, n[1], c, f0, Rn[i]) #d0
    #             alpha_E_g = Utils.compute_path_loss_coefficient(drx_ris_E, n[2], c, f0, Rn[i]) #d0
                
    #         sigma_gE_sq = alpha_E_g**2 * (1 / (2 * (K_E_g + 1))) # np.sqrt(alpha_E_g / 2) *
         
    #         sigma_e_sq = sigma_gE_sq * NEEV
            
    #         if channel_model == "rician":
    #             channel_H = np.zeros((N, K), dtype=np.complex128)
    #             for k in range(K):
    #                 channel_H[:, k] = alpha_h[k] * Utils.generate_rician_channel(N, K_h).flatten()
    #             channel_G_B = alpha_B_g * Utils.generate_rician_channel(NR_B * N, K_B_g).reshape(NR_B, N)
    #             channel_gE = alpha_E_g * Utils.generate_rician_channel(N * NR_E, K_E_g).reshape(N, NR_E)

    #         elif channel_model == "rayleigh":
    #             channel_H = np.array([alpha_h[k] * Utils.generate_rayleigh_channel(N) for k in range(K)]).T
    #             channel_G_B = alpha_B_g * Utils.generate_rayleigh_channel(NR_B * N).reshape(NR_B, N)
    #             channel_gE = alpha_E_g * Utils.generate_rayleigh_channel(N * NR_E).reshape(N, NR_E)

    #         elif channel_model == "nakagami":
    #             m = 1  # Nakagami shape parameter, can be adjusted
    #             channel_H = np.array([alpha_h[k] * Utils.generate_nakagami_channel(N, m) for k in range(K)]).T
    #             channel_G_B = alpha_B_g * Utils.generate_nakagami_channel(NR_B * N, m).reshape(NR_B, N)
    #             channel_gE = alpha_E_g * Utils.generate_nakagami_channel(N * NR_E, m).reshape(N, NR_E)

    #         else:
    #             raise ValueError("Unsupported channel model. Choose from 'rician', 'rayleigh', 'nakagami'.")

    #         channel_gE_error =  Utils.generate_cscg_channel(N, sigma_e_sq)

    #         channel_G_B_lst.append(channel_G_B)
    #         channel_gE_lst.append(channel_gE)
    #         channel_gE_error_lst.append(channel_gE_error)
    #         channel_H_lst.append(channel_H)
            
    #         # sigma_e_sq_lst.append(sigma_e_sq)

    #     return channel_G_B_lst, channel_gE_lst, channel_gE_error_lst, channel_H_lst, sigma_e_sq 

    @staticmethod
    def dbm_to_watts(power_dbm):
        """
        Convert power from dBm to watts.

        Parameters:
        - power_dbm (float): Power in dBm.

        Returns:
        - float: Power in watts.
        """
        return 10 ** ((power_dbm - 30) / 10)

    @staticmethod
    def set_power_range(power_min_dbm, power_max_dbm, power_step_dbm):
        """
        Generate power range in dBm and watts.

        Parameters:
        - power_min_dbm (float): Minimum power in dBm.
        - power_max_dbm (float): Maximum power in dBm.
        - power_step_dbm (float): Step size for power in dBm.

        Returns:
        - tuple: Power range in dBm and watts.
        """
        power_range_dbm = np.arange(power_min_dbm, power_max_dbm + power_step_dbm, power_step_dbm)
        power_range_watts = Utils.dbm_to_watts(power_range_dbm)
        return power_range_dbm, power_range_watts

    @staticmethod
    def calculate_noise_power_variance(N0_dBm, BW, NF):
        """
        Calculate the noise power variance.

        Parameters:
        - N0_dBm (float): Noise power spectral density in dBm/Hz.
        - BW (float): Bandwidth in Hz.
        - NF (float): Noise figure in dB.

        Returns:
        - float: Noise power variance.
        """
        N0 = 10 ** ((N0_dBm - 30) / 10)
        return N0 * BW * (10 ** (NF / 10))

    @staticmethod
    def compute_Pc(P0, Pcn, P0_RIS, N):
        """
        Compute the total power consumption.

        Parameters:
        - P0 (float): Static power consumption of all other nodes in dBm.
        - Pcn (float): Static power consumption per RIS element in dBm.
        - P0_RIS (float): Static power consumption of the RIS in dBm.
        - N (int): Number of RIS elements.

        Returns:
        - float: Total power consumption in watts.
        """
        P0_W = 10 ** ((P0 - 30) / 10)
        Pcn_W = 10 ** ((Pcn - 30) / 10)
        P0_RIS_W = 10 ** ((P0_RIS - 30) / 10)
        return P0_W + N * Pcn_W + P0_RIS_W

    @staticmethod
    def compute_R(H, p, sigma_RIS_sq):
        """
        Compute the R matrix for the RIS.

        Parameters:
        - H (np.ndarray): Channel matrix UE-RIS.
        - p (np.ndarray): Power allocation vector.
        - sigma_RIS_sq (float): Noise variance at the RIS.

        Returns:
        - np.ndarray: Computed R matrix.
        """
        K = H.shape[1]
        N = H.shape[0]
        R = np.zeros((N, N), dtype=complex)
        for k in range(K):
            H_k = np.diag(H[:, k])
            R += p[k] * (H_k.conj().T @ H_k)
        R += sigma_RIS_sq * np.eye(N)
        return R

    @staticmethod
    def compute_Ptot_active_algo1(R, gamma, p, mu, Pc, ris_state):
        """
        Compute the total power consumption for active RIS algorithm.

        Parameters:
        - R (np.ndarray): R matrix.
        - gamma (np.ndarray): RIS reflection coefficients.
        - p (np.ndarray): Power allocation vector.
        - mu (float): Amplifier inefficiency.
        - Pc (float): Static power consumption.
        - ris_state (str): RIS state ('active' or 'passive').

        Returns:
        - float: Total power consumption.
        """
        ris_bool = 1 if ris_state == 'active' else 0
        N = gamma.shape[0]
        return ris_bool * np.real(np.trace(((gamma @ gamma.conj().T) - np.eye(N)) @ R)) + np.sum(mu * p) + Pc

    @staticmethod
    def LMMSE_receiver_active_Bob(G, H, gamma, p, sigma_sq, sigma_RIS_sq):
        """
        Compute the LMMSE receiver for Bob.

        Parameters:
        - G (np.ndarray): Channel matrix for Bob.
        - H (np.ndarray): Channel matrix UE-RIS.
        - gamma (np.ndarray): RIS reflection coefficients.
        - p (np.ndarray): Power allocation vector.
        - sigma_sq (float): Noise variance.
        - sigma_RIS_sq (float): RIS noise variance.

        Returns:
        - np.ndarray: LMMSE receiver for Bob.
        """
        K = p.shape[0]
        NR = G.shape[0]
        C_active = np.zeros((NR, K), dtype=np.complex128)
        Gamma = np.diagflat(gamma)
        W = sigma_sq * np.eye(NR) + sigma_RIS_sq * G @ (Gamma @ Gamma.conj().T) @ G.conj().T
        for k in range(K):
            hk = H[:, k]
            Hk = np.diag(hk)
            Ak = G @ Hk
            interf_k = 0
            for m in range(K):
                if m != k:
                    hm = H[:, m]
                    Hm = np.diag(hm)
                    Am = G @ Hm
                    interf_k += p[m] * Am @ (gamma @ gamma.conj().T) @ Am.conj().T
            Mk = interf_k + W
            Akg = Ak @ gamma
            C_active[:, k] = np.sqrt(p[k]) * np.linalg.inv(Mk) @ Akg.flatten()
        return C_active

    @staticmethod
    def LMMSE_receiver_active_Eve(G, H, gamma, p, sigma_sq, sigma_RIS_sq):
        """
        Compute the LMMSE receiver for Eve.

        Parameters:
        - G (np.ndarray): Channel matrix for Eve.
        - H (np.ndarray): Channel matrix.
        - gamma (np.ndarray): RIS reflection coefficients.
        - p (np.ndarray): Power allocation vector.
        - sigma_sq (float): Noise variance.
        - sigma_RIS_sq (float): RIS noise variance.

        Returns:
        - np.ndarray: LMMSE receiver for Eve.
        """
        K = p.shape[0]
        NR = G.shape[1]
        C_active = np.zeros((NR, K), dtype=np.complex128)
        Gamma = np.diagflat(gamma)
        W = sigma_sq + sigma_RIS_sq * G.conj().T @ (Gamma.conj().T @ Gamma) @ G
        for k in range(K):
            hk = H[:, k]
            Hk = np.diag(hk)
            Ak = G.conj().T @ Hk
            interf_k = 0
            for m in range(K):
                if m != k:
                    hm = H[:, m]
                    Hm = np.diag(hm)
                    Am = G.conj().T @ Hm
                    interf_k += p[m] * Am @ (gamma @ gamma.conj().T) @ Am.conj().T
            Mk = interf_k + W
            Akg = Ak @ gamma
            C_active[:, k] = np.sqrt(p[k]) * np.linalg.inv(Mk) @ Akg.flatten()
        return C_active

    @staticmethod
    def sinr_active_Bob(C, G, H, gamma, p, sigma_sq, sigma_RIS_sq):
        """
        Compute the SINR for Bob.

        Parameters:
        - C (np.ndarray): LMMSE receiver matrix for Bob.
        - G (np.ndarray): Channel matrix for Bob.
        - H (np.ndarray): Channel matrix.
        - gamma (np.ndarray): RIS reflection coefficients.
        - p (np.ndarray): Power allocation vector.
        - sigma_sq (float): Noise variance.
        - sigma_RIS_sq (float): RIS noise variance.

        Returns:
        - np.ndarray: SINR for Bob.
        """
        epsilon = np.finfo(float).eps
        K = p.shape[0]
        NR = G.shape[0]
        sinr_a = np.zeros_like(p)
        for k in range(K):
            ck = C[:, k].reshape(NR, 1)
            hk = H[:, k]
            Hk = np.diag(hk)
            Ak = G @ Hk
            num_k = p[k] * np.sum(np.abs(ck.conj().T @ Ak @ gamma) ** 2)
            interf_m = 0
            for m in range(K):
                if m != k:
                    hm = H[:, m]
                    Hm = np.diag(hm)
                    Am = G @ Hm
                    interf_m += p[m] * np.sum(np.abs(ck.conj().T @ Am @ gamma) ** 2)
            uk = G.conj().T @ ck
            Uk_tilt = np.diagflat(np.abs(uk) ** 2)
            noise_k = sigma_sq * np.linalg.norm(ck) ** 2 + sigma_RIS_sq * np.linalg.norm(sqrtm(Uk_tilt) @ gamma)**2
            denom_k = noise_k + interf_m
            sinr_a[k] = num_k / np.where(denom_k > epsilon, denom_k, epsilon)
        return sinr_a

    @staticmethod
    def sinr_active_Eve(G, H, gamma, p, sigma_sq, sigma_RIS_sq, sigma_e_sq, scsi_bool):
        """
        Compute the SINR for Eve.

        Parameters:
        - G (np.ndarray): Channel matrix for Eve.
        - H (np.ndarray): Channel matrix.
        - gamma (np.ndarray): RIS reflection coefficients.
        - p (np.ndarray): Power allocation vector.
        - sigma_sq (float): Noise variance.
        - sigma_RIS_sq (float): RIS noise variance.
        - sigma_g_sq (float): Variance of the channel estimation error.
        - scsi_bool (bool): Flag for imperfect CSI.

        Returns:
        - np.ndarray: SINR for Eve.
        """
        epsilon = np.finfo(float).eps
        K = p.shape[0]
        N = H.shape[0]
        sinr_a = np.zeros_like(p)
        RE = G @ G.conj().T + scsi_bool * sigma_e_sq * np.eye(N)
        for k in range(K):
            hk = H[:, k]
            Hk = np.diag(hk)
            num_k = p[k] * np.linalg.norm(sqrtm(RE) @ Hk @ gamma) ** 2
            interf_m = 0
            for m in range(K):
                if m != k:
                    hm = H[:, m]
                    Hm = np.diag(hm)
                    interf_m += p[m] * np.linalg.norm(sqrtm(RE) @ Hm @ gamma) ** 2
            noise_k = sigma_sq + sigma_RIS_sq * np.linalg.norm(sqrtm(RE) @ gamma)**2
            denom_k = noise_k + interf_m
            sinr_a[k] = num_k / np.where(denom_k > epsilon, denom_k, epsilon)
        return sinr_a

    @staticmethod
    def sinr_active_Eve_orig(G, H, gamma, p, sigma_sq, sigma_RIS_sq):
        """
        Compute the original SINR for Eve without imperfect CSI.

        Parameters:
        - G (np.ndarray): Channel matrix for Eve.
        - H (np.ndarray): Channel matrix.
        - gamma (np.ndarray): RIS reflection coefficients.
        - p (np.ndarray): Power allocation vector.
        - sigma_sq (float): Noise variance.
        - sigma_RIS_sq (float): RIS noise variance.

        Returns:
        - np.ndarray: SINR for Eve.
        """
        epsilon = np.finfo(float).eps
        K = p.shape[0]
        N = H.shape[0]
        sinr_a = np.zeros_like(p)
        Gamma = np.diagflat(gamma)
        for k in range(K):
            hk = H[:, k]
            Hk = np.diag(hk)
            num_k = p[k] * np.sum(np.abs(G.conj().T @ Hk @ gamma)**2)
            interf_m = 0
            for m in range(K):
                if m != k:
                    hm = H[:, m]
                    Hm = np.diag(hm)
                    interf_m += p[m] * np.sum(np.abs(G.conj().T @ Hm @ gamma)**2)
            noise_k = sigma_sq + sigma_RIS_sq * np.real(G.conj().T @ (Gamma @ Gamma.conj().T) @ G)
            denom_k = noise_k + interf_m
            sinr_a[k] = num_k / np.where(denom_k > epsilon, denom_k, epsilon)
        return sinr_a

    @staticmethod
    def SR_active_algo1(G, H, gamma, p, sigma_sq, sigma_RIS_sq, sigma_e_sq, scsi_bool, orig_bool, Rx):
        """
        Compute the sum-rate for active RIS algorithm 1.

        Parameters:
        - G (np.ndarray): Channel matrix for the receiver.
        - H (np.ndarray): Channel matrix.
        - gamma (np.ndarray): RIS reflection coefficients.
        - p (np.ndarray): Power allocation vector.
        - sigma_sq (float): Noise variance.
        - sigma_RIS_sq (float): RIS noise variance.
        - sigma_g_sq (float): Variance of the channel estimation error.
        - scsi_bool (bool): Flag for imperfect CSI.
        - orig_bool (bool): Flag for original SINR calculation.
        - Rx (str): Receiver type ('Bob' or 'Eve').

        Returns:
        - float: Sum-rate.
        """
        K = p.shape[0]
        if Rx == "Bob":
            C = Utils.LMMSE_receiver_active_Bob(G, H, gamma, p, sigma_sq, sigma_RIS_sq)
            sinr = Utils.sinr_active_Bob(C, G, H, gamma, p, sigma_sq, sigma_RIS_sq)
        else:
            if not orig_bool:
                sinr = Utils.sinr_active_Eve(G, H, gamma, p, sigma_sq, sigma_RIS_sq, sigma_e_sq, scsi_bool)
            else:
                sinr = Utils.sinr_active_Eve_orig(G, H, gamma, p, sigma_sq, sigma_RIS_sq)
        return sum(np.log2(1 + sinr[k]) for k in range(K))

    @staticmethod
    def GEE_active_algo1(G, H, gamma, p, mu, Pc, sigma_sq, sigma_RIS_sq, sigma_e_sq, ris_state, scsi_bool, orig_bool, Rx):
        """
        Compute the generalized energy efficiency for active RIS algorithm 1.

        Parameters:
        - G (np.ndarray): Channel matrix for the receiver.
        - H (np.ndarray): Channel matrix.
        - gamma (np.ndarray): RIS reflection coefficients.
        - p (np.ndarray): Power allocation vector.
        - mu (float): Amplifier inefficiency.
        - Pc (float): Static power consumption.
        - sigma_sq (float): Noise variance.
        - sigma_RIS_sq (float): RIS noise variance.
        - sigma_g_sq (float): Variance of the channel estimation error.
        - ris_state (str): RIS state ('active' or 'passive').
        - scsi_bool (bool): Flag for imperfect CSI.
        - orig_bool (bool): Flag for original SINR calculation.
        - Rx (str): Receiver type ('Bob' or 'Eve').

        Returns:
        - float: Generalized energy efficiency.
        """
        sr_algo1 = Utils.SR_active_algo1(G, H, gamma, p, sigma_sq, sigma_RIS_sq, sigma_e_sq, scsi_bool, orig_bool, Rx)
        R = Utils.compute_R(H, p, sigma_RIS_sq)
        Ptot = Utils.compute_Ptot_active_algo1(R, gamma, p, mu, Pc, ris_state)
        return sr_algo1 / Ptot

    @staticmethod
    def save_output_data(names, *args):
        """
        Save output data with given names.

        Parameters:
        - names (list): List of names for the data.
        - args (list): Data to be saved.

        Returns:
        - dict: Dictionary of saved data.
        """
        if len(names) != len(args):
            raise ValueError("The number of names must match the number of arguments passed.")
        return {name: arg for name, arg in zip(names, args)}

    @staticmethod
    def create_data_saver():
        """
        Create a data saver function that can save and reset output data.

        Returns:
        - callable: Data saver function.
        - callable: Reset function.
        """
        output_arrays = {}

        def save_output_data(names, *args):
            if len(names) != len(args):
                raise ValueError("The number of names must match the number of arguments passed.")
            for name, arg in zip(names, args):
                if name in output_arrays:
                    output_arrays[name].extend(arg)
                else:
                    output_arrays[name] = list(arg)
            return output_arrays

        def reset_data():
            current_data = dict(output_arrays)
            output_arrays.clear()
            return current_data

        return save_output_data, reset_data

    @staticmethod
    def flatten_list_of_dicts(dict_list):
        """
        Flatten a list of dictionaries into a single dictionary.

        Parameters:
        - dict_list (list): List of dictionaries.

        Returns:
        - dict: Flattened dictionary.
        """
        combined_dict = {}
        for single_dict in dict_list:
            for key, value in single_dict.items():
                if key not in combined_dict:
                    combined_dict[key] = [value]
                else:
                    combined_dict[key].append(value)
        return combined_dict

    @staticmethod
    def flatten_nested_list_of_dicts(nested_dict_list):
        """
        Flatten a nested list of dictionaries into a single dictionary.

        Parameters:
        - nested_dict_list (list): Nested list of dictionaries.

        Returns:
        - dict: Flattened dictionary.
        """
        combined_dict = {}
        for dict_list in nested_dict_list:
            if not all(isinstance(item, dict) for item in dict_list):
                raise ValueError("Each item in nested_dict_list must be a list of dictionaries.")
            for single_dict in dict_list:
                for key, value in single_dict.items():
                    if key not in combined_dict:
                        combined_dict[key] = [value]
                    else:
                        combined_dict[key].append(value)
        return combined_dict

    @staticmethod
    def flatten_and_group_nested_list_of_dicts(nested_dict_list):
        """
        Flatten and group a nested list of dictionaries into a single dictionary.

        Parameters:
        - nested_dict_list (list): Nested list of dictionaries.

        Returns:
        - dict: Flattened and grouped dictionary.
        """
        combined_dict = {}
        for dict_list in nested_dict_list:
            temp_dict = {}
            for single_dict in dict_list:
                for key, value in single_dict.items():
                    if key not in temp_dict:
                        temp_dict[key] = [value]
                    else:
                        temp_dict[key].append(value)
            for key, value_list in temp_dict.items():
                if key not in combined_dict:
                    combined_dict[key] = [value_list]
                else:
                    combined_dict[key].append(value_list)
        return combined_dict
    


    @staticmethod
    def flatten_and_group_nested_list_of_dicts_ver2(nested_dict_list):
        """
        Flatten and group a nested list of dictionaries into a single dictionary,
        grouping by sample_index and ordering by params_combo values.

        Parameters:
        - nested_dict_list (list): Nested list of dictionaries.

        Returns:
        - dict: Flattened and grouped dictionary.
        """
        combined_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for result in nested_dict_list:
            sample_index = result.get('sample_index')
            if sample_index is None:
                continue

            params_combo = result.get('params_combo')
            if params_combo is None:
                continue

            params_keys = tuple(params_combo.keys())
            params_values = tuple(params_combo.values())

            for key, value in result.items():
                if key not in ['sample_index', 'params_combo']:
                    combined_dict[sample_index][params_keys][params_values].append((key, value))

        final_result = {}
        for sample_index, params_dict in combined_dict.items():
            final_result[sample_index] = {}
            for params_keys, nested_dict in params_dict.items():
                if len(params_keys) == 1:  # Single parameter case
                    param_values = sorted(nested_dict.keys())
                    final_result[sample_index][params_keys] = [v[0] for v in param_values]
                    for param_value in param_values:
                        for key, value in nested_dict[param_value]:
                            if key not in final_result[sample_index]:
                                final_result[sample_index][key] = []
                            final_result[sample_index][key].append(value)
                else:  # Multi-parameter case
                    final_result[sample_index][params_keys] = {}
                    outer_param_dict = defaultdict(lambda: defaultdict(list))
                    for param_values, param_data in nested_dict.items():
                        outer_param = param_values[0]
                        inner_params = param_values[1:]
                        outer_param_dict[outer_param][inner_params].extend(param_data)
                    
                    for outer_param, inner_dict in outer_param_dict.items():
                        inner_values_sorted = sorted(inner_dict.keys())
                        final_result[sample_index][params_keys][outer_param] = {'Ptmax': [iv[0] for iv in inner_values_sorted]}
                        for inner_values in inner_values_sorted:
                            for key, value in inner_dict[inner_values]:
                                if key not in final_result[sample_index][params_keys][outer_param]:
                                    final_result[sample_index][params_keys][outer_param][key] = []
                                final_result[sample_index][params_keys][outer_param][key].append(value)

        return final_result

    @staticmethod
    def average_values_by_key_equal_length(flattened_dict, num_samples):
        """
        Average values by key in a flattened dictionary.

        Parameters:
        - flattened_dict (dict): Flattened dictionary.
        - num_samples (int): Number of samples.

        Returns:
        - dict: Dictionary with averaged values.
        """
        keys_to_average = {
            "sr_uniform_Bob_pcsi", "sr_uniform_Bob_scsi", "sr_uniform_Eve_pcsi",
            "sr_uniform_Eve_scsi", "ssr_uniform_pcsi", "ssr_uniform_scsi", "gee_uniform_Bob_pcsi", "gee_uniform_Bob_scsi",
            "gee_uniform_Eve_pcsi", "gee_uniform_Eve_scsi", "see_uniform_pcsi", "see_uniform_scsi", "sr_sol_pcsi", "sr_sol_Q_pcsi", "ssr_sol_pcsi", "ssr_sol_Q_pcsi",
            "sr_sol_scsi", "sr_sol_Q_scsi", "ssr_sol_scsi", "ssr_sol_Q_scsi", "gee_sol_pcsi", "gee_sol_Q_pcsi", "see_sol_pcsi", "see_sol_Q_pcsi", "gee_sol_scsi", "gee_sol_Q_scsi", "see_sol_scsi", "see_sol_Q_scsi", "iteration_altopt_pcsi", "iteration_altopt_scsi",
            "iteration_p_pcsi", "iteration_p_scsi", "iteration_gamma_pcsi", "iteration_gamma_scsi",
            "time_complexity_altopt_pcsi", "time_complexity_altopt_scsi", "time_complexity_p_pcsi",
            "time_complexity_p_scsi", "time_complexity_gamma_pcsi", "time_complexity_gamma_scsi"
        }

        avg_dict = {}
        for i in range(num_samples):
            for key, value_lists in flattened_dict[i].items():
                if key in keys_to_average:
                    if key in {"ssr_sol_Q_pcsi", "ssr_sol_Q_scsi", "see_sol_Q_pcsi", "see_sol_Q_scsi"}:
                        # Special case: list of dictionaries
                        grouped_dict = {}
                        for i, sample_dicts in enumerate(value_lists):
                            for sample_dict in sample_dicts:
                                for k, v in sample_dict.items():
                                    if k not in grouped_dict.keys():
                                        grouped_dict[k] = [[] for _ in range(len(value_lists))]
                                    grouped_dict[k][i].append(v)
    
                        avg_dict_i = {}
                        for k, values_list in grouped_dict.items():
                            avg_list_i = [sum(values) / num_samples for values in zip(*values_list)]
                            avg_dict_i[k] = avg_list_i
                        
                        avg_dict[key] = avg_dict_i
                    else:
                        # General case
                        avg_list = [sum(values) / num_samples for values in zip(*value_lists)]
                        avg_dict[key] = avg_list
                else:
                    avg_dict[key] = value_lists  # Preserve original values for keys not in keys_to_average

        return avg_dict


    # @staticmethod
    # def average_values_by_key_equal_length(flattened_dict):
    #     """
    #     Average values by key in a flattened dictionary.

    #     Parameters:
    #     - flattened_dict (dict): Flattened dictionary.

    #     Returns:
    #     - dict: Dictionary with averaged values.
    #     """
    #     avg_dict = {}
    #     for key, value_lists in flattened_dict.items():
    #         num_lists = len(value_lists)
    #         zipped_values = zip(*value_lists)
    #         avg_list = [sum(values) / num_lists for values in zipped_values]
    #         avg_dict[key] = avg_list
    #     return avg_dict
    
    @staticmethod
    def average_lists(lists: List[List[float]]) -> List[float]:
        """
        Averages the values in a list of lists element-wise.
        """
        if not lists:
            return []
        
        # Debug statement to check the input
        # print(f"Averaging lists: {lists}")

        averaged = [sum(values) / len(values) for values in zip(*lists)]
        return averaged

    @staticmethod
    def average_nested_dicts(dicts: List[Dict]) -> Dict:
        """
        Averages the values in a list of nested dictionaries element-wise.
        """
        if not dicts:
            return {}
        
        keys = dicts[0].keys()

        # Debug statement to check the input
        # print(f"Averaging nested dictionaries for keys: {keys}")

        averaged = {}
        for key in keys:
            values = [d[key] for d in dicts]
            # Ensure the values are lists before averaging
            if all(isinstance(v, (int, float)) for v in values):
                averaged[key] = sum(values) / len(values)
            else:
                averaged[key] = Utils.average_lists(values)
        return averaged

    
    def average_results(results: Dict[int, Dict[str, Any]], keys_to_average: set) -> Dict[str, Any]:
        """
        Averages the key-value list across the sample indexes element-wise,
        while maintaining the structure and leaving the parts not inside the
        selected key names unchanged.
        """
        averaged_results = defaultdict(dict)
        
        for key in results[0].keys():
            if key in keys_to_average:
                # Gather the lists to average
                lists_to_average = [results[idx][key] for idx in results]
                
                # Debug statement to check the input
                # print(f"Processing key: {key}, lists to average: {lists_to_average}")

                # Determine if the elements are lists of numbers or lists of dictionaries
                if all(isinstance(item, (int, float)) for sublist in lists_to_average for item in sublist):
                    averaged_results[key] = Utils.average_lists(lists_to_average)
                elif all(isinstance(item, dict) for sublist in lists_to_average for item in sublist):
                    # Group values by their keys across all sub-lists
                    grouped_values = defaultdict(lambda: [[] for _ in range(len(lists_to_average))])
                    for idx, sublist in enumerate(lists_to_average):
                        for d in sublist:
                            for k, v in d.items():
                                grouped_values[k][idx].append(v)
                    
                    # Convert grouped_values to lists of lists for averaging
                    averaged_results[key] = {k: Utils.average_lists(v) for k, v in grouped_values.items()}
                else:
                    print(f"Unhandled data structure for key: {key}")
            else:
                # Copy the values directly (unchanged)
                for idx in results:
                    averaged_results[idx][key] = results[idx][key]
        
        return averaged_results

    @staticmethod
    def load_and_access_results(file_path):
        """
        Load and access results from a file.

        Parameters:
        - file_path (str): Path to the file.

        Returns:
        - dict: Loaded results.
        """
        with np.load(file_path, allow_pickle=True) as data:
            return data['arr_0'].item()

    @staticmethod
    def generate_cscg_channel(N, sigma_g_squared):
        """
        Generate a (N x 1) Circularly Symmetric Complex Gaussian (CSCG) channel vector.

        Parameters:
        - N (int): Number of entries in the channel vector.
        - sigma_g_squared (float): Variance of the CSCG distribution.

        Returns:
        - np.ndarray: A (N x 1) complex array representing the CSCG channel.
        """
        sigma = np.sqrt(sigma_g_squared / 2)
        real_part = np.random.normal(loc=0, scale=sigma, size=N)
        imaginary_part = np.random.normal(loc=0, scale=sigma, size=N)
        return (real_part + 1j * imaginary_part).reshape(N, 1)

    @staticmethod
    def calculate_ris_surface_size(N_x, N_y, wavelength, spacing):
        """
        Calculate the total size of a 2D RIS surface.

        Parameters:
        - N_x (int): Number of unit cells in the x-direction.
        - N_y (int): Number of unit cells in the y-direction.
        - wavelength (float): Wavelength of the operating frequency.
        - spacing (float): Spacing between unit cells.

        Returns:
        - float: Total size of the 2D RIS surface.
        """
        unit_cell_size = wavelength / 2 + spacing
        L_x = N_x * unit_cell_size
        L_y = N_y * unit_cell_size
        return L_x * L_y

    @staticmethod
    def calculate_near_field_distance_from_area(A_RIS, wavelength):
        """
        Calculate the near-field distance of a Reconfigurable Intelligent Surface (RIS).

        Parameters:
        - A_RIS (float): Area of the RIS surface in square meters.
        - wavelength (float): Wavelength of the operating frequency in meters.

        Returns:
        - float: Near-field distance of the RIS in meters.
        """
        D = np.sqrt(A_RIS)
        return 2 * (D ** 2) / wavelength

    @staticmethod
    def calculate_near_field_distance(D, wavelength):
        """
        Calculate the near-field distance of a Reconfigurable Intelligent Surface (RIS).

        Parameters:
        - D (float): Diameter of the RIS.
        - wavelength (float): Wavelength of the operating frequency.

        Returns:
        - float: Near-field distance of the RIS.
        """
        return 2 * D**2 / wavelength

    @staticmethod
    def generate_rician_channel(N, K_factor):
        """
        Generate a Rician fading channel vector.

        Parameters:
        - N (int): Number of entries in the channel vector.
        - K_factor (float): Rician K-factor.

        Returns:
        - np.ndarray: A (N x 1) complex array representing the Rician fading channel.
        """
        s = np.sqrt(K_factor / (K_factor + 1))
        sigma = np.sqrt(1 / (2 * (K_factor + 1)))
        real_part = s + sigma * np.random.randn(N)
        imaginary_part = sigma * np.random.randn(N)
        return (real_part + 1j * imaginary_part).reshape(N, 1)

    @staticmethod
    def generate_rayleigh_channel(N):
        """
        Generate a Rayleigh fading channel vector.

        Parameters:
        - N (int): Number of entries in the channel vector.

        Returns:
        - np.ndarray: A (N x 1) complex array representing the Rayleigh fading channel.
        """
        real_part = np.random.randn(N)
        imaginary_part = np.random.randn(N)
        return (real_part + 1j * imaginary_part) / np.sqrt(2)

    @staticmethod
    def generate_nakagami_channel(N, m):
        """
        Generate a Nakagami fading channel vector.

        Parameters:
        - N (int): Number of entries in the channel vector.
        - m (float): Shape parameter of the Nakagami distribution.

        Returns:
        - np.ndarray: A (N x 1) complex array representing the Nakagami fading channel.
        """
        magnitude = np.random.gamma(shape=m, scale=1/m, size=N)
        phase = np.random.uniform(0, 2 * np.pi, size=N)
        return magnitude * (np.cos(phase) + 1j * np.sin(phase))

    @staticmethod
    def apply_shadowing(distance, mean_db=0, std_db=8):
        """
        Apply shadowing effect to the path loss.

        Parameters:
        - distance (float): Distance over which shadowing is applied (not directly used but included for completeness).
        - mean_db (float): Mean value of shadowing in dB.
        - std_db (float): Standard deviation of shadowing in dB.

        Returns:
        - float: Shadowing effect in linear scale.
        """
        shadowing_db = np.random.normal(mean_db, std_db)
        return 10 ** (shadowing_db / 10)

    @staticmethod
    def generate_positions(K, r_cell, h_BS, h_RIS, hmin_UE, hmax_UE,  Rn_B, Rn_RIS, vic_percent_eve):
        """
        Generate positions for the RIS, base station (Bob), Eve, and UEs.

        Parameters:
        - K (int): Number of UEs.
        - r_cell (float): Radius of the cell.
        - h_BS (float): Height of the base station.
        - h_RIS (float): Height of the RIS.
        - hmin_UE (float): Minimum height of the UEs.
        - hmax_UE (float): Maximum height of the UEs.
        - Rn (float): Near-field distance of the RIS.

        Returns:
        - tuple: Positions of the RIS, base station (Bob), Eve, and UEs.
        """
        r_vicinity =  vic_percent_eve * (r_cell - Rn_RIS)
        RIS_pos = np.array([r_cell, r_cell, h_RIS])
        Rx_B = np.array([r_cell + r_vicinity + Rn_RIS, r_cell, h_BS])
        dist_ris_bob = np.linalg.norm([Rx_B[0] - RIS_pos[0], Rx_B[1] - RIS_pos[1]])
        

        while True:
            x_E, y_E = np.random.uniform(0, 2 * r_cell, 2)
            dist_ris_eve = np.linalg.norm([x_E - RIS_pos[0], y_E - RIS_pos[1]])
            dist_bob_eve = np.linalg.norm([x_E - Rx_B[0], y_E - Rx_B[1]])
            if Rn_RIS <= dist_ris_eve <= r_cell and  Rn_B <= dist_bob_eve <= r_vicinity:  # Rn < dist_bob_eve < r_cell:
                break

        Rx_E = np.array([x_E, y_E, np.random.uniform(hmin_UE, hmax_UE)]) # hmax_UE, h_BS
        Tx = np.zeros((K, 3))

        for i in range(K):
            while True:
                x, y = np.random.uniform(0, 2 * r_cell, 2)
                dist_ris_ue = np.linalg.norm([x - RIS_pos[0], y - RIS_pos[1]])
                if dist_ris_ue >= max(Rn_RIS, Rn_B):
                    break
            z = np.random.uniform(hmin_UE, hmax_UE)
            Tx[i] = [x, y, z]

        return RIS_pos, Rx_B, Rx_E, Tx

    @staticmethod
    def compute_path_length(point1, point2):
        """
        Compute the Euclidean distance between two points.

        Parameters:
        - point1 (np.ndarray): First point.
        - point2 (np.ndarray): Second point.

        Returns:
        - float: Euclidean distance.
        """
        return np.sqrt(np.sum((point1 - point2)**2))

    @staticmethod
    def compute_path_loss_coefficient(d, n, c, f0, d0):
        """
        Compute the path loss coefficient.

        Parameters:
        - d (float): Distance.
        - n (float): Path loss exponent.
        - c (float): Speed of light.
        - f0 (float): Frequency.
        - d0 (float): Reference distance.

        Returns:
        - float: Path loss coefficient.
        """
        PLo = (4 * np.pi * f0 * d0 / c)**(-2)
        return np.sqrt(2 * PLo) / np.sqrt(1 + (d / d0)**(n))
    
    @staticmethod
    def nearest_value(quantization_set, value):
        """
        Find the nearest value in the quantization set to the given value.
        Parameters:
            quantization_set (np.array): The set of quantized values.
            value (float): The value to be quantized.
        Returns:
            float: The nearest value in the quantization set.
        """
        return quantization_set[np.argmin(np.abs(quantization_set - value))]

    @staticmethod
    def project_to_quantized_levels(gamma, amin, amax, bits_phase, bits_amplitude):
        """
        Project the complex reflection coefficients to the nearest quantized values for both amplitude and phase.
        Parameters:
            gamma (np.array): N*1 complex numpy array vector of reflection coefficients.
            amax (float): Maximum amplitude value.
            bits_phase (int): Number of bits for quantizing the phase.
            bits_amplitude (int): Number of bits for quantizing the amplitude.
        Returns:
            np.array: N*1 complex numpy array vector of quantized reflection coefficients.
        """
        
        N = gamma.shape[0]

        # Calculate the number of quantization levels for phase and amplitude
        Q_phase = 2 ** bits_phase
        Q_amplitude = 2 ** bits_amplitude

        # Create the quantization sets for phase and amplitude
        phase_set = np.linspace(0, 2 * np.pi, Q_phase, endpoint=False)
        amplitude_set = np.linspace(amin, amax, Q_amplitude, endpoint=True)

        # Extract amplitude and phase from gamma
        amplitudes = np.abs(gamma).reshape((N,))
        phases = np.angle(gamma).reshape((N,))

        phases = (phases + 2 * np.pi) % (2 * np.pi)

        # Initialize quantized amplitude and phase arrays
        quantized_amplitudes = np.array([Utils.nearest_value(amplitude_set, amp) for amp in amplitudes])
        quantized_phases = np.array([Utils.nearest_value(phase_set, phase) for phase in phases])

        # Reconstruct the quantized gamma vector
        quantized_gamma = quantized_amplitudes * np.exp(1j * quantized_phases)

        return quantized_gamma.reshape((N, 1))
    
    @staticmethod
    def get_unit_and_explanation(param):
        if param == "N":
            return "", "Number of RIS reflecting elements"
        elif param == "Ptmax":
            return "dBm", "Combined Maximum Transmit Power of all the UEs"
        elif param == "PRmax":
            return "dBm", "Maximum Radio Frequency Power of the RIS"
        elif param == "a":
            return "dB", "Gain of the RIS"
        elif param == "Pcn":
            return "dBm", "Power allocated to each RIS element to enable reconfiguration"
        elif param == "NEEV":
            return "dB", "Normalized Error Variance for Eve"
        else:
            return "", "Unknown parameter"
    
    # Function to determine N_x and N_y such that N_x * N_y = N and N_x >= N_y
    @staticmethod
    def compute_N_x_y(N):
        """
        Function to compute N_x and N_y such that N_x * N_y = N.

        Parameters:
        - N (int): The number of RIS elements.

        Returns:
        - tuple: A tuple containing (N_x, N_y).
        """
        sqrt_N = int(np.sqrt(N))
        for factor in range(sqrt_N, 0, -1):
            if N % factor == 0:
                return factor, N // factor
        return N, 1
    

    @staticmethod
    def check_X_feasibility(
        X: np.ndarray,
        R: np.ndarray,
        Rnorm: np.ndarray,
        ris_state: str,
        rf_state: str,
        cons_state: str,
        PRmax_val: float,
        tol: float = 1e-6
    ) -> Tuple[bool, Dict]:
        """
        Check whether a candidate X (N×N Hermitian) satisfies:
          1.  X ⪰ 0 (up to small negative eigenvalues ≥ -tol).
          2.  If ris_state == 'active' and rf_state == 'RF-Power':
                trace(R @ X) ∈ [ trace(R) - tol,  trace(R) + PRmax_val + tol ]
              If ris_state == 'active' and rf_state == 'RF-Gain':
                trace(Rnorm @ X) ∈ [ trace(Rnorm) - tol,  trace(Rnorm) + PRmax_val + tol ]
              If ris_state == 'inactive' and cons_state == 'global':
                trace(Rnorm @ X) ≤ trace(Rnorm) + tol
              If ris_state == 'inactive' and cons_state != 'global':
                diag(X) ≤ 1 + tol
        Returns:
          overall_feasible: bool        (True if all checks pass)
          info: dict                    (detailed status for each check)
        """

        info: Dict[str, float] = {}

        # 1) Symmetrize X to guard against tiny numerical asymmetry
        X_sym = (X + X.conj().T) / 2

        # 1.a) Check Hermitian (off‐diagonal difference small)
        herm_diff = np.linalg.norm(X - X.conj().T, ord='fro')
        info['hermitian_error'] = float(herm_diff)

        # 1.b) Check PSD: eigenvalues of X_sym should be >= -tol
        eigvals = np.linalg.eigvalsh(X_sym)
        min_eig = np.min(eigvals)
        is_PSD = (min_eig >= -tol)
        info['min_eigenvalue'] = float(min_eig)
        info['is_PSD'] = float(is_PSD)

        # Precompute traces (using X_sym, the symmetrized version)
        trace_R_X     = float(np.real(np.trace(R @ X_sym)))
        trace_Rnorm_X = float(np.real(np.trace(Rnorm @ X_sym)))
        trace_R       = float(np.real(np.trace(R)))
        trace_Rnorm   = float(np.real(np.trace(Rnorm)))

        info['trace(R X)']     = trace_R_X
        info['trace(R)']       = trace_R
        info['trace(Rnorm X)'] = trace_Rnorm_X
        info['trace(Rnorm)']   = trace_Rnorm

        # Decide which linear constraints to check
        feas_lin = True

        if ris_state.lower() == 'active':
            if rf_state == 'RF-Power':
                low_bound  = trace_R - tol
                high_bound = trace_R + PRmax_val + tol
                ok_low     = (trace_R_X >= low_bound)
                ok_high    = (trace_R_X <= high_bound)
                info['RF-Power lower_ok']   = float(ok_low)
                info['RF-Power upper_ok']   = float(ok_high)
                info['RF-Power lower_bound'] = low_bound
                info['RF-Power upper_bound'] = high_bound
                feas_lin &= (ok_low and ok_high)

            elif rf_state == 'RF-Gain':
                low_bound  = trace_Rnorm - tol
                high_bound = trace_Rnorm + PRmax_val + tol
                ok_low     = (trace_Rnorm_X >= low_bound)
                ok_high    = (trace_Rnorm_X <= high_bound)
                info['RF-Gain lower_ok']     = float(ok_low)
                info['RF-Gain upper_ok']     = float(ok_high)
                info['RF-Gain lower_bound']  = low_bound
                info['RF-Gain upper_bound']  = high_bound
                feas_lin &= (ok_low and ok_high)

            else:
                # Unexpected rf_state
                info['error'] = f"Unknown rf_state = {rf_state!r}"
                feas_lin = False

        else:  # ris_state != 'active'
            if cons_state == 'global':
                upper_bound = trace_Rnorm + tol
                ok = (trace_Rnorm_X <= upper_bound)
                info['global_consistency_ok'] = float(ok)
                info['global_upper_bound']    = upper_bound
                feas_lin &= ok

            else:
                diag_X = np.real(np.diag(X_sym))
                max_diag = float(np.max(diag_X))
                ok = np.all(diag_X <= 1.0 + tol)
                info['diag_max']      = max_diag
                info['diag_upper_ok'] = float(ok)
                feas_lin &= ok

        # Combine PSD + linear feasibility
        overall_feasible = bool(is_PSD and feas_lin)

        # Print a human‐readable summary
        print("=== check_X_feasibility summary ===")
        print(f"  Hermitian‐error (‖X – X^H‖_F): {herm_diff:.3e}   (tol = {tol})")
        print(f"  Minimum eigenvalue            : {min_eig:.3e}   => PSD? {is_PSD}")
        if ris_state.lower() == 'active':
            if rf_state == 'RF-Power':
                print(
                    f"  trace(R X) = {trace_R_X:.3e},  "
                    f"bounds = [ {trace_R:.3e}, {trace_R + PRmax_val:.3e} ]  "
                    f"=> low_ok={ok_low}, high_ok={ok_high}"
                )
            else:  # RF-Gain
                print(
                    f"  trace(Rnorm X) = {trace_Rnorm_X:.3e},  "
                    f"bounds = [ {trace_Rnorm:.3e}, {trace_Rnorm + PRmax_val:.3e} ]  "
                    f"=> low_ok={ok_low}, high_ok={ok_high}"
                )
        else:
            if cons_state == 'global':
                print(
                    f"  trace(Rnorm X) = {trace_Rnorm_X:.3e},  "
                    f"upper_bound = {trace_Rnorm:.3e}  => ok={ok}"
                )
            else:
                print(
                    f"  max(diag(X)) = {max_diag:.3e},  "
                    f"upper_bound = 1.0  => ok={ok}"
                )
        print(f"  => Overall feasible? {overall_feasible}")
        print("====================================\n")

        return overall_feasible, info

    
    # def compute_N_x_y(N):
    #     N_x = int(np.ceil(np.sqrt(N)))
    #     N_y = N // N_x
    #     while N_x * N_y != N:
    #         N_x -= 1
    #         N_y = N // N_x
    #     return N_x, N_y

    
