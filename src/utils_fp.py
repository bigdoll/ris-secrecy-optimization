import numpy as np
from scipy.linalg import sqrtm
from visualization import *
import os

def check_file_exists(file_path):
    """
    Check if a file exists in the given path.

    Parameters:
    - file_path (str): Path of the file to check.

    Returns:
    - exists (bool): Flag indicating if the file exists.
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

def get_file_path(file_name):
    file_path = os.path.join("./data/samples", file_name)
    return file_path

def check_channel_or_ris_file_exists(generate_samples_func):
    def wrapper_generate_samples(*args, **kwargs):
        filename = args[1] if args else kwargs.get('filename')
        if not filename:
            raise ValueError("Filename is missing.")
        file_path = get_file_path(filename)
        if os.path.exists(file_path):
            samples = np.load(file_path, allow_pickle=True)
            print("Samples loaded from file.")
        else:
            samples = generate_samples_func(*args, **kwargs)
            np.save(file_path, samples)
            print("Samples saved to file.")
        return samples
    return wrapper_generate_samples

@check_channel_or_ris_file_exists
def monte_carlo_simulation(num_samples, filename, *args, **kwargs):
    channel_samples = []
    for _ in range(num_samples):
        channels = generate_channels_algo2(*args, **kwargs)
        channel_samples.append(channels)
    channel_samples = create_object_array_from_tuples(channel_samples)
    return channel_samples

@check_channel_or_ris_file_exists
def generate_ris_coefficients(num_samples, filename, N_range, channel_samples_range, *args, **kwargs):
    """
    Generate random RIS reflection coefficients for RIS and save them in a file.

    Args:
    - num_samples (int): Number of samples to generate.
    - filename (str): Name of the file to save the samples.
    - N_range (list): Range of RIS elements.
    - channel_samples_range (list): Range of channel samples.

    Returns:
    - np.ndarray: Generated RIS coefficients.
    """
    ris_coefficients_samples = []

    for i in range(num_samples):
        ris_coefficients_lst = []
        G_B_lst, gE_hat_lst, gE_error_lst, H_lst = channel_samples_range[i]

        for n_index, N in enumerate(N_range):
            while True:
                ris_coefficients = np.exp(1j * 2 * np.pi * np.random.rand(N, 1))
                if kwargs['state'] == 'passive':
                    ssr_ris = np.random.rand()  # Placeholder for actual SSR computation
                    if ssr_ris > 0:
                        ris_coefficients_samples.append(ris_coefficients)
                        break
                else:
                    sr_ris_Bob = SR_active_algo1(G_B_lst[n_index], H_lst[n_index], ris_coefficients, *args, scsi_bool=0, orig_bool=True, Rx="Bob")
                    sr_ris_Eve = SR_active_algo1(gE_hat_lst[n_index] + gE_error_lst[n_index], H_lst[n_index], ris_coefficients, *args, scsi_bool=0, orig_bool=True, Rx="Eve")
                    ssr_ris = sr_ris_Bob - sr_ris_Eve
                    if ssr_ris > 0:
                        ris_coefficients_lst.append(ris_coefficients)
                        break
        ris_coefficients_samples.append(ris_coefficients_lst)

    return create_object_array_from_tuples(ris_coefficients_samples)

def generate_channels_algo2(r_cell, h_BS, h_RIS, hmin_UE, hmax_UE, d0, N_range, K, NR, f0, lambda_0, d, c, n, R_K, sigma_g_sq, channel_model="rician", shadowing=False):
    """
    Generate channel samples for the simulation with optional shadowing effects.

    Parameters:
    - r_cell, h_BS, h_RIS, hmin_UE, hmax_UE, d0, N_range, K, NR, f0, lambda_0, d, c, n, R_K, sigma_g_sq: Simulation parameters.
    - channel_model (str): Channel model to use ("rician", "rayleigh", "nakagami").
    - shadowing (bool): Whether to include shadowing effects.

    Returns:
    - tuple: Generated channel samples.
    """
    channel_G_B_lst = []
    channel_gE_hat_lst = []
    channel_gE_error_lst = []
    channel_H_lst = []

    N_max = np.max(N_range)
    N_x = N_y = int(np.sqrt(N_max))
    A_RIS = calculate_ris_surface_size(N_x, N_y, wavelength=lambda_0, spacing=d)
    Rn = calculate_near_field_distance_from_area(A_RIS, wavelength=lambda_0)
    RIS_pos, Rx_B, Rx_E, Tx = generate_positions(K, r_cell, h_BS, h_RIS, hmin_UE, hmax_UE, Rn)
    
    plot_3d_positions(RIS_pos, Tx, Rx_B, Rx_E)

    dtx_ris = np.array([compute_path_length(RIS_pos, tx) for tx in Tx])
    drx_ris_B = compute_path_length(Rx_B, RIS_pos)
    drx_ris_E = compute_path_length(Rx_E, RIS_pos)

    if shadowing:
        alpha_h = np.array([compute_path_loss_coefficient(d, n[0], c, f0, d0) * apply_shadowing(d) for d in dtx_ris])
        alpha_B_g = compute_path_loss_coefficient(drx_ris_B, n[1], c, f0, d0) * apply_shadowing(drx_ris_B)
        alpha_E_g = compute_path_loss_coefficient(drx_ris_E, n[2], c, f0, d0) * apply_shadowing(drx_ris_E)
    else:
        alpha_h = compute_path_loss_coefficient(dtx_ris, n[0], c, f0, d0)
        alpha_B_g = compute_path_loss_coefficient(drx_ris_B, n[1], c, f0, d0)
        alpha_E_g = compute_path_loss_coefficient(drx_ris_E, n[2], c, f0, d0)

    K_h, K_B_g, K_E_g = R_K
    NR_B, NR_E = NR

    for N in N_range:
        if channel_model == "rician":
            channel_H = np.zeros((N, K), dtype=np.complex128)
            for k in range(K):
                channel_H[:, k] = alpha_h[k] * generate_rician_channel(N, K_h).flatten()
            channel_G_B = alpha_B_g * generate_rician_channel(NR_B * N, K_B_g).reshape(NR_B, N)
            channel_gE_hat = alpha_E_g * generate_rician_channel(N * NR_E, K_E_g).reshape(N, NR_E)

        elif channel_model == "rayleigh":
            channel_H = np.array([alpha_h[k] * generate_rayleigh_channel(N) for k in range(K)]).T
            channel_G_B = alpha_B_g * generate_rayleigh_channel(NR_B * N).reshape(NR_B, N)
            channel_gE_hat = alpha_E_g * generate_rayleigh_channel(N * NR_E).reshape(N, NR_E)

        elif channel_model == "nakagami":
            m = 1  # Nakagami shape parameter, can be adjusted
            channel_H = np.array([alpha_h[k] * generate_nakagami_channel(N, m) for k in range(K)]).T
            channel_G_B = alpha_B_g * generate_nakagami_channel(NR_B * N, m).reshape(NR_B, N)
            channel_gE_hat = alpha_E_g * generate_nakagami_channel(N * NR_E, m).reshape(N, NR_E)

        else:
            raise ValueError("Unsupported channel model. Choose from 'rician', 'rayleigh', 'nakagami'.")

        channel_gE_error = generate_cscg_channel(N, sigma_g_sq)

        channel_G_B_lst.append(channel_G_B)
        channel_gE_hat_lst.append(channel_gE_hat)
        channel_gE_error_lst.append(channel_gE_error)
        channel_H_lst.append(channel_H)

    return channel_G_B_lst, channel_gE_hat_lst, channel_gE_error_lst, channel_H_lst

def create_object_array_from_tuples(channel_samples):
    samples_array = np.empty(len(channel_samples), dtype=object)
    for i, sample in enumerate(channel_samples):
        samples_array[i] = sample
    return samples_array

def dbm_to_watts(power_dbm):
    power_watts = 10 ** ((power_dbm - 30) / 10)
    return power_watts

def set_power_range(power_min_dbm, power_max_dbm, power_step_dbm):
    power_range_dbm = np.arange(power_min_dbm, power_max_dbm + power_step_dbm, power_step_dbm)
    power_range_watts = dbm_to_watts(power_range_dbm)
    return power_range_dbm, power_range_watts

def calculate_noise_power_variance(N0_dBm, BW, NF):
    N0 = 10 ** ((N0_dBm - 30) / 10)
    sigma_sq = N0 * BW * (10 ** (NF / 10))
    return sigma_sq

def compute_Pc(P0, Pcn, P0_RIS, N):
    P0_W = 10 ** ((P0 - 30) / 10)
    Pcn_W = 10 ** ((Pcn - 30) / 10)
    P0_RIS_W = 10 ** ((P0_RIS - 30) / 10)
    Pc = P0_W + N * Pcn_W + P0_RIS_W
    return Pc

def compute_R(H, p, sigma_RIS_sq):
    K = H.shape[1]
    N = H.shape[0]
    R = np.zeros((N, N), dtype=complex)
    for k in range(K):
        H_k = np.diag(H[:, k])
        R += p[k] * (H_k.conj().T @ H_k)
    R += sigma_RIS_sq * np.eye(N)
    return R

def compute_Ptot_active_algo1(R, gamma, p, mu, Pc, ris_state):
    ris_bool = 1 if ris_state == 'active' else 0
    N = gamma.shape[0]
    Ptot = ris_bool * np.real(np.trace(((gamma @ gamma.conj().T) - np.eye(N)) @ R)) + np.sum(mu * p) + Pc
    return Ptot

def LMMSE_receiver_active_Bob(G, H, gamma, p, sigma_sq, sigma_RIS_sq):
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

def LMMSE_receiver_active_Eve(G, H, gamma, p, sigma_sq, sigma_RIS_sq):
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

def sinr_active_Bob(C, G, H, gamma, p, sigma_sq, sigma_RIS_sq):
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

def sinr_active_Eve(G, H, gamma, p, sigma_sq, sigma_RIS_sq, sigma_g_sq, scsi_bool):
    epsilon = np.finfo(float).eps
    K = p.shape[0]
    N = H.shape[0]
    sinr_a = np.zeros_like(p)
    RE = G @ G.conj().T + scsi_bool * sigma_g_sq*np.eye(N)
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

def sinr_active_Eve_orig(G, H, gamma, p, sigma_sq, sigma_RIS_sq):
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

def SR_active_algo1(G, H, gamma, p, sigma_sq, sigma_RIS_sq, sigma_g_sq, scsi_bool, orig_bool, Rx):
    K = p.shape[0]
    if Rx == "Bob":
        C = LMMSE_receiver_active_Bob(G, H, gamma, p, sigma_sq, sigma_RIS_sq)
        sinr = sinr_active_Bob(C, G, H, gamma, p, sigma_sq, sigma_RIS_sq)
    else:
        if not orig_bool:
            sinr = sinr_active_Eve(G, H, gamma, p, sigma_sq, sigma_RIS_sq, sigma_g_sq, scsi_bool)
        else:
            sinr = sinr_active_Eve_orig(G, H, gamma, p, sigma_sq, sigma_RIS_sq)
    sr_algo1 = sum(np.log2(1 + sinr[k]) for k in range(K))
    return sr_algo1

def GEE_active_algo1(G, H, gamma, p, mu, Pc, sigma_sq, sigma_RIS_sq, sigma_g_sq, ris_state, scsi_bool, orig_bool, Rx):
    sr_algo1 = SR_active_algo1(G, H, gamma, p, sigma_sq, sigma_RIS_sq, sigma_g_sq, scsi_bool, orig_bool, Rx)
    R = compute_R(H, p, sigma_RIS_sq)
    Ptot = compute_Ptot_active_algo1(R, gamma, p, mu, Pc, ris_state)
    gee_algo1 = sr_algo1 / Ptot
    return gee_algo1

def save_output_data(names, *args):
    if len(names) != len(args):
        raise ValueError("The number of names must match the number of arguments passed.")
    output_arrays = {name: arg for name, arg in zip(names, args)}
    return output_arrays

def create_data_saver():
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

def flatten_list_of_dicts(dict_list):
    combined_dict = {}
    for single_dict in dict_list:
        for key, value in single_dict.items():
            if key not in combined_dict:
                combined_dict[key] = [value]
            else:
                combined_dict[key].append(value)
    return combined_dict

def flatten_nested_list_of_dicts(nested_dict_list):
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

def flatten_and_group_nested_list_of_dicts(nested_dict_list):
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

def average_values_by_key_equal_length(flattened_dict):
    avg_dict = {}
    for key, value_lists in flattened_dict.items():
        num_lists = len(value_lists)
        zipped_values = zip(*value_lists)
        avg_list = [sum(values) / num_lists for values in zipped_values]
        avg_dict[key] = avg_list
    return avg_dict

def load_and_access_results(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        result_dict = data['arr_0'].item()
    return result_dict

def generate_cscg_channel(N, sigma_g_squared):
    sigma = np.sqrt(sigma_g_squared / 2)
    real_part = np.random.normal(loc=0, scale=sigma, size=N)
    imaginary_part = np.random.normal(loc=0, scale=sigma, size=N)
    cscg_channel = (real_part + 1j * imaginary_part).reshape(N, 1)
    return cscg_channel

def calculate_ris_surface_size(N_x, N_y, wavelength, spacing):
    """
    Calculate the total size of a 2D RIS surface.

    Parameters:
        N_x (int): Number of unit cells in the x-direction.
        N_y (int): Number of unit cells in the y-direction.
        wavelength (float): Wavelength of the operating frequency.
        spacing (float): Spacing between unit cells.

    Returns:
        ris_surface_size (float): Total size of the 2D RIS surface.
    """
    unit_cell_size = wavelength / 2 + spacing
    L_x = N_x * unit_cell_size
    L_y = N_y * unit_cell_size
    ris_surface_size = L_x * L_y
    return ris_surface_size

def calculate_near_field_distance_from_area(A_RIS, wavelength):
    """
    Calculate the near-field distance of a Reconfigurable Intelligent Surface (RIS)
    when the area of the RIS is given.

    Parameters:
        A_RIS (float): Area of the RIS surface in square meters.
        wavelength (float): Wavelength of the operating frequency in meters.

    Returns:
        near_field_distance (float): Near-field distance of the RIS in meters.
    """
    D = np.sqrt(A_RIS)
    R_rayleigh = 2 * (D ** 2) / wavelength
    near_field_distance = R_rayleigh
    return near_field_distance

def calculate_near_field_distance(D, wavelength):
    return 2 * D**2 / wavelength

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
    cscg_channel = (real_part + 1j * imaginary_part).reshape(N, 1)
    return cscg_channel

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
    rician_channel = (real_part + 1j * imaginary_part).reshape(N, 1)
    return rician_channel
  
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
    rayleigh_channel = (real_part + 1j * imaginary_part) / np.sqrt(2)
    return rayleigh_channel

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
    nakagami_channel = magnitude * (np.cos(phase) + 1j * np.sin(phase))
    return nakagami_channel

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
    shadowing_linear = 10 ** (shadowing_db / 10)
    return shadowing_linear

def generate_positions(K, r_cell, h_BS, h_RIS, hmin_UE, hmax_UE, Rn):
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
    RIS_pos = np.array([r_cell, r_cell, h_RIS])
    Rx_B = np.array([r_cell + Rn, r_cell, h_BS])

    while True:
        x_E, y_E = np.random.uniform(0, 2 * r_cell, 2)
        dist_bob_eve = np.linalg.norm([x_E - Rx_B[0], y_E - Rx_B[1]])
        if Rn < dist_bob_eve < r_cell:
            break

    Rx_E = np.array([x_E, y_E, np.random.uniform(hmin_UE, hmax_UE)])
    Tx = np.zeros((K, 3))

    for i in range(K):
        while True:
            x, y = np.random.uniform(0, 2 * r_cell, 2)
            dist_ris_ue = np.linalg.norm([x - r_cell, y - r_cell])
            if dist_ris_ue >= Rn:
                break
        z = np.random.uniform(hmin_UE, hmax_UE)
        Tx[i] = [x, y, z]

    return RIS_pos, Rx_B, Rx_E, Tx

def compute_path_length(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def compute_path_loss_coefficient(d, n, c, f0, d0):
    PLo = (4 * np.pi * f0 * d0 / c)**(-2)
    alpha = np.sqrt(2 * PLo) / np.sqrt(1 + (d / d0)**(n))
    return alpha
