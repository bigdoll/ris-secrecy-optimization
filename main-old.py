import sys
import os

# Add the 'src' directory to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from multiprocessing import Pool, cpu_count
from config import SystemConfig
from utils import Utils
from gamma_utils import GammaUtils
from power_utils import PowerUtils
from optimizers import Optimizer

# Initialize system configuration
config = SystemConfig()

# Initialize other utility classes
# utils = Utils()


# Constants and configuration from SystemConfig
N_RANGE = config.N_RANGE
NUM_SAMPLES = config.NUM_SAMPLES
PTMAX_DBM = config.PTMAX_DBM
# PTMAX = config.PTMAX
PTMIN = config.PTMIN
P_TEMP = (PTMIN / config.K) * np.ones(config.K)
FILENAME_CHANNEL = config.FILENAME_CHANNEL
FILENAME_RIS = config.FILENAME_RIS
OUTPUT_FILE = config.OUTPUT_FILE
NAMES = config.NAMES

# Compute noise power
if config.ris_state == 'active':
    sigma_sq = sigma_RIS_sq = config.sigma_RIS_sq
else:
    sigma_sq = config.sigma_RIS_sq
    sigma_RIS_sq = 0

# sigma_g_sq = config.sigma_g_sq
# SNR_E = config.SNR_E
Pc_passive = [config.compute_static_power_consumption(N) for N in N_RANGE]
Pc_active = [config.compute_static_power_consumption(N) for N in N_RANGE]
Pc = Pc_active if config.ris_state == 'active' else Pc_passive

# Generate channel and RIS samples
channel_samples_range = Utils.monte_carlo_simulation(
    NUM_SAMPLES, FILENAME_CHANNEL, config.r_cell, config.h_BS, config.h_RIS, config.hmin_UE, config.hmax_UE, config.d0, N_RANGE, config.K, config.NR, config.f0, config.lambda_0, config.d, config.c, config.n, config.R_K, config.NEEV, config.vic_percent_eve, channel_model="rician", shadowing=False
)

ris_samples_range = Utils.generate_ris_coefficients(
    NUM_SAMPLES, FILENAME_RIS, N_RANGE, channel_samples_range, P_TEMP, sigma_sq, sigma_RIS_sq, scsi_bool=0, algo="algo1", state='active'
)

output_data = []

def simulation(sample_index):
    print(f'\nStarting Simulation for sample_index: {sample_index}\n')
    G_B_lst, gE_hat_lst, gE_error_lst, H_lst, sigma_e_sq = channel_samples_range[sample_index]
    ris_samples = ris_samples_range[sample_index]

    for n_index, N in enumerate(N_RANGE):
        print(f'\nRunning Simulation! for n_index: {n_index}, Number of RIS elements: {N}\n')
        G_B = G_B_lst[n_index]
        gE_hat = gE_hat_lst[n_index]
        gE_error = gE_error_lst[n_index]
        H = H_lst[n_index]
        # sigma_e_sq = sigma_e_sq_lst[n_index]
        gE = gE_hat + gE_error
        gamma = ris_samples[n_index]
        
        # gamma_utils = GammaUtils(H, G_B, gE_hat, sigma_sq, sigma_RIS_sq, sigma_e_sq, config.mu, Pc[n_index], scsi_bool=1)
        # power_utils = PowerUtils()

        for p_index, Ptmax in enumerate(config.power_range_watts):  # Loop for each maximum power budget
            # PRmax = config.q * Ptmax
            
            print(f'\n******************** power_index: {p_index}, Maximum available Transmit Power in dBm: {config.power_range_dbm[p_index]} dBm ********************\n')
            
            
            p_uniform = (Ptmax / config.K) * np.ones(config.K)
            R = Utils.compute_R(H, p_uniform, sigma_RIS_sq)
            
            # Ensure R is positive definite
            assert np.all(np.linalg.eigvals(R) > 0), "Matrix R is not positive definite."

            # Compute the eigenvalues of R
            eigenvalues_R = np.linalg.eigvalsh(R)

            # Find the minimum and maximum eigenvalue
            min_eigenvalue_R = np.min(eigenvalues_R)
            max_eigenvalue_R = np.max(eigenvalues_R)

            if config.ris_state == 'active':
                PRmax = (config.a - 1)* np.real(np.trace(R))  if config.rf_state == 'RF-Gain'else config.PRmax
            else:
                PRmax = 0
            
            optimizer_psci = Optimizer(H, G_B, gE_hat, gE_error, sigma_sq, sigma_RIS_sq, sigma_e_sq, config.mu, Pc[n_index], PRmax, config.a, Ptmax, config.BW, scsi_bool=0) # PRmax
            
            optimizer_scsi = Optimizer(H, G_B, gE_hat, gE_error, sigma_sq, sigma_RIS_sq, sigma_e_sq, config.mu, Pc[n_index], PRmax, config.a, Ptmax, config.BW, scsi_bool=1) # PRmax
            
            rho_norm = np.sqrt(
            (PRmax + np.real(np.trace(R))) / np.real(np.sum(gamma.conj().T @ R @ gamma)))  if config.cons_state == 'global' else 1
            
            rho_min = np.sqrt((PRmax + np.real(np.trace(R))) / (N * max_eigenvalue_R)) if config.cons_state == 'global' else 1
            
            rho_max = np.sqrt((PRmax + np.real(np.trace(R))) / (N * min_eigenvalue_R)) if config.cons_state == 'global' else 1
          
            # gamma_random = rho_max * gamma #if config.ris_state == 'active' else gamma
            
            gamma_random = rho_norm * gamma
            
            # p_init_pcsi = p_init_scsi = p_uniform
            # gamma_init_pcsi = gamma_init_scsi = gamma_random
                
            # Initialization
            if p_index == 0:
                # Uniform power allocation - initialization
                p_init_pcsi = p_init_scsi = p_uniform
                gamma_init_pcsi = gamma_init_scsi = gamma_random

            else:
                # Warm start with previous solution
                p_init_pcsi = p_sol_pcsi
                gamma_init_pcsi = gamma_sol_pcsi
                p_init_scsi = p_sol_scsi
                gamma_init_scsi = gamma_sol_scsi

            # Compute sum rate for uniform power and random RIS allocation
            sr_uniform_Bob_pcsi = Utils.SR_active_algo1(G_B, H, gamma_random, p_uniform, sigma_sq, sigma_RIS_sq, sigma_e_sq, scsi_bool=0, orig_bool=True, Rx="Bob")
            sr_uniform_Eve_pcsi = Utils.SR_active_algo1(gE, H, gamma_random, p_uniform, sigma_sq, sigma_RIS_sq, sigma_e_sq, scsi_bool=0, orig_bool=True, Rx="Eve")
            ssr_uniform_pcsi = max(sr_uniform_Bob_pcsi - sr_uniform_Eve_pcsi, 0)

            sr_uniform_Bob_scsi = Utils.SR_active_algo1(G_B, H, gamma_random, p_uniform, sigma_sq, sigma_RIS_sq, sigma_e_sq, scsi_bool=0, orig_bool=True, Rx="Bob")
            sr_uniform_Eve_scsi = Utils.SR_active_algo1(gE_hat, H, gamma_random, p_uniform, sigma_sq, sigma_RIS_sq, sigma_e_sq, scsi_bool=1, orig_bool=False, Rx="Eve")
            ssr_uniform_scsi = max(sr_uniform_Bob_scsi - sr_uniform_Eve_scsi, 0)

            # Compute global energy efficiency for uniform power and random RIS allocation
            gee_uniform_Bob_pcsi = config.BW * Utils.GEE_active_algo1(G_B, H, gamma_random, p_uniform, config.mu, Pc[n_index], sigma_sq, sigma_RIS_sq, sigma_e_sq, config.ris_state, scsi_bool=0, orig_bool=True, Rx="Bob")
            gee_uniform_Eve_pcsi = config.BW * Utils.GEE_active_algo1(gE, H, gamma_random, p_uniform, config.mu, Pc[n_index], sigma_sq, sigma_RIS_sq, sigma_e_sq, config.ris_state, scsi_bool=0, orig_bool=True, Rx="Eve")
            see_uniform_pcsi = max(gee_uniform_Bob_pcsi - gee_uniform_Eve_pcsi, 0)

            gee_uniform_Bob_scsi = config.BW * Utils.GEE_active_algo1(G_B, H, gamma_random, p_uniform, config.mu, Pc[n_index], sigma_sq, sigma_RIS_sq, sigma_e_sq, config.ris_state, scsi_bool=0, orig_bool=True, Rx="Bob")
            gee_uniform_Eve_scsi = config.BW * Utils.GEE_active_algo1(gE_hat, H, gamma_random, p_uniform, config.mu, Pc[n_index], sigma_sq, sigma_RIS_sq, sigma_e_sq, config.ris_state, scsi_bool=1, orig_bool=False, Rx="Eve")
            see_uniform_scsi = max(gee_uniform_Bob_scsi - gee_uniform_Eve_scsi, 0)

            # Secrecy Optimization
            print(f"\npCSI: Starting {config.ris_state.capitalize()} Secrecy {config.opt_state.upper()} optimization...\n")
            p_sol_pcsi, gamma_sol_pcsi, gamma_sol_Q_pcsi, ssr_sol_pcsi, ssr_sol_Q_pcsi, see_sol_pcsi, see_sol_Q_pcsi, iteration_altopt_pcsi, iteration_p_pcsi, iteration_gamma_pcsi, time_complexity_altopt_pcsi, time_complexity_p_pcsi, time_complexity_gamma_pcsi = optimizer_psci.altopt_algo1_mod(
                gamma_init_pcsi, p_init_pcsi, config.opt_bool[config.opt_state], config.ris_state, config.cons_state, config.bits_range, quantization=True
            )

            print(f"\nsCSI: Starting {config.ris_state.capitalize()} Secrecy {config.opt_state.upper()} optimization...\n")
            p_sol_scsi, gamma_sol_scsi, gamma_sol_Q_scsi, ssr_sol_scsi, ssr_sol_Q_scsi, see_sol_scsi, see_sol_Q_scsi, iteration_altopt_scsi, iteration_p_scsi, iteration_gamma_scsi, time_complexity_altopt_scsi, time_complexity_p_scsi, time_complexity_gamma_scsi = optimizer_scsi.altopt_algo1_mod(
                gamma_init_scsi, p_init_scsi, config.opt_bool[config.opt_state], config.ris_state, config.cons_state, config.bits_range, quantization=True
            )

            print(f"\nStoring output values for (sample Index: {sample_index}, N_Index: {n_index}, No RIS element: {N}, P_Index: {p_index}, Max Transmit power : {config.power_range_dbm[p_index]} dBm)...\n")
            output_data.append(Utils.save_output_data(
                NAMES, sample_index, n_index, N, p_index, Ptmax, p_uniform, gamma_random, p_sol_pcsi, p_sol_scsi, gamma_sol_pcsi, gamma_sol_Q_pcsi, gamma_sol_scsi, gamma_sol_Q_scsi, sr_uniform_Bob_pcsi, sr_uniform_Bob_scsi, sr_uniform_Eve_pcsi, sr_uniform_Eve_scsi, ssr_uniform_pcsi, ssr_uniform_scsi, gee_uniform_Bob_pcsi, gee_uniform_Bob_scsi, gee_uniform_Eve_pcsi, gee_uniform_Eve_scsi, see_uniform_pcsi, see_uniform_scsi, ssr_sol_pcsi, ssr_sol_Q_pcsi, ssr_sol_scsi, ssr_sol_Q_scsi, see_sol_pcsi, see_sol_Q_pcsi, see_sol_scsi,see_sol_Q_scsi,
                iteration_altopt_pcsi, iteration_altopt_scsi, iteration_p_pcsi, iteration_p_scsi, iteration_gamma_pcsi, iteration_gamma_scsi, time_complexity_altopt_pcsi, time_complexity_altopt_scsi, time_complexity_p_pcsi, time_complexity_p_scsi, time_complexity_gamma_pcsi, time_complexity_gamma_scsi))

    return output_data

def run_simulation():
    print("################################# STARTING SIMULATION: LETS GO! #################################")
    num_processes = cpu_count()
    print(f"Number of CPU cores available: {num_processes}")

    with Pool(processes=num_processes) as pool:
        output_results = pool.map(simulation, range(NUM_SAMPLES))

    return output_results

if __name__ == '__main__':
    results = run_simulation()
    output_data = Utils.flatten_and_group_nested_list_of_dicts(results)
    np.savez_compressed(OUTPUT_FILE, output_data)
    print("******************************************** SIMULATION SUCCESSFULLY COMPLETED! CIAO 😇 ********************************************")
