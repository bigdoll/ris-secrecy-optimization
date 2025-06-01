import sys
import os

# Add the 'src' directory to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import product
from config import SystemConfig
from utils import Utils
from gamma_utils import GammaUtils
from power_utils import PowerUtils
from optimizers import Optimizer

def introduce_problem():
    print("""
    ##############################################################
    #    Resource Allocation and Energy Efficiency Management    #
    #      in RIS-Aided Physical Layer Security Communication     #
    ##############################################################
    """)
    print("""
    This project addresses the problem of optimizing resource allocation and energy efficiency in RIS-aided uplink communication systems with a focus on physical layer security. We consider different simulation scenarios to evaluate the impact of various system parameters on the overall performance.
    """)

def prompt_user_for_config(config):
    use_default = input("Do you want to use the default configuration? (yes/no): ").strip().lower()
    
    if use_default == 'no':
        config.rf_state = input("Enter RF state ('RF-Gain' or 'RF-Power'): ").strip().lower()
        config.ris_state = input("Enter RIS state ('active' or 'passive'): ").strip().lower()
        config.cons_state = input("Enter constraint state ('global' or 'local'): ").strip().lower()
        config.opt_state = input("Enter optimization state ('sr' or 'ee'): ").strip().lower()
        config.quantization = True if input("Do you want to enable quantization? (yes/no): ").strip().lower() == 'yes' else False
        config.bits_range = eval(input("Enter bits range as a list of tuples, e.g., [(1,1), (2,2), (3,3), (4,4)]: "))
        
        config.K = int(input("Enter number of UEs: "))
        config.NR = [int(x) for x in input("Enter number of receive antennas at the BS and Eve (comma-separated): ").strip().split(',')]
        config.BW = float(input("Enter system bandwidth in Hz: "))
        config.f0 = float(input("Enter operating frequency in Hz: "))
        config.r_cell = float(input("Enter cell radius in meters: "))
        config.h_BS = float(input("Enter height of the BS in meters: "))
        config.h_RIS = float(input("Enter height of the RIS in meters: "))
        config.hmin_UE = float(input("Enter minimum height for UEs in meters: "))
        config.hmax_UE = float(input("Enter maximum height for UEs in meters: "))
        config.NUM_SAMPLES = int(input("Enter number of samples: "))
        
        fixed_N = int(input("Enter fixed value for N: "))
        fixed_Ptmax = float(input("Enter fixed value for Ptmax (dBm): "))
        fixed_PRmax = float(input("Enter fixed value for PRmax (dBm): "))
        fixed_a = float(input("Enter fixed value for a (dB): "))
        fixed_Pcn_p = float(input("Enter fixed value for Pcn_p (dBm): "))
        fixed_Pcn_a = float(input("Enter fixed value for Pcn_a (dBm): "))
    
        config.update_custom_params(N=fixed_N, Ptmax=fixed_Ptmax, PRmax=fixed_PRmax, a=fixed_a, Pcn_p=fixed_Pcn_p, Pcn_a=fixed_Pcn_a)

    return config

def select_simulation_type(config):
    print("""
    Select the type of simulation to run:
    1. SSR, SEE Vs Ptmax
    2. SSR, SEE Vs N
    3. SSR, SEE Vs PRmax
    4. SSR, SEE Vs a
    5. SSR, SEE Vs Pcn
    6. Customize (Select multiple parameters to vary)
    """)
    simulation_type = int(input("Enter the number corresponding to the simulation type: "))
    config.update_output_file(simulation_type)
    return simulation_type, config

def configure_simulation_params(simulation_type, config):
    if simulation_type == 1:
        ptmax_min = float(input("Enter minimum Ptmax (in dBm): "))
        ptmax_max = float(input("Enter maximum Ptmax (in dBm): "))
        ptmax_step = float(input("Enter Ptmax step (in dBm): "))
        config.varying_params = {'Ptmax': np.arange(ptmax_min, ptmax_max + ptmax_step, ptmax_step)}
        config.run_separately = True
    elif simulation_type == 2:
        n_min = int(input("Enter minimum N (number of RIS elements): "))
        n_max = int(input("Enter maximum N (number of RIS elements): "))
        n_step = int(input("Enter N step: "))
        config.varying_params = {'N': range(n_min, n_max + n_step, n_step)}
        config.run_separately = True
    elif simulation_type == 3:
        PR_min = float(input("Enter minimum PR_min (in dBm): "))
        PR_max = float(input("Enter maximum PR_max (in dBm): "))
        PR_step = float(input("Enter a step (in dBm): "))
        config.varying_params = {'PRmax': np.arange(PR_min, PR_max + PR_step, PR_step)}
        config.run_separately = True
    elif simulation_type == 4:
        a_min = float(input("Enter minimum a (in dB): "))
        a_max = float(input("Enter maximum a (in dB): "))
        a_step = float(input("Enter a step (in dB): "))
        config.varying_params = {'a': np.arange(a_min, a_max + a_step, a_step)}
        config.run_separately = True
    elif simulation_type == 5:
        pcn_min = float(input("Enter minimum Pcn (in dBm): "))
        pcn_max = float(input("Enter maximum Pcn (in dBm): "))
        pcn_step = float(input("Enter Pcn step (in dBm): "))
        config.varying_params = {'Pcn': np.arange(pcn_min, pcn_max + pcn_step, pcn_step)}
        config.run_separately = True
    elif simulation_type == 6:
        config.varying_params = {}
        print("Enter the parameters you want to vary (comma-separated): Ptmax, N, a, Pcn")
        params = input().strip().split(',')
        for param in params:
            param = param.strip()
            if param == 'Ptmax':
                ptmax_min = float(input("Enter minimum Ptmax (in dBm): "))
                ptmax_max = float(input("Enter maximum Ptmax (in dBm): "))
                ptmax_step = float(input("Enter Ptmax step (in dBm): "))
                config.varying_params['Ptmax'] = np.arange(ptmax_min, ptmax_max + ptmax_step, ptmax_step)
            elif param == 'N':
                n_min = int(input("Enter minimum N (number of RIS elements): "))
                n_max = int(input("Enter maximum N (number of RIS elements): "))
                n_step = int(input("Enter N step: "))
                config.varying_params['N'] = range(n_min, n_max + n_step, n_step)
            elif param == 'PRmax':
                PR_min = float(input("Enter minimum PR_min (in dBm): "))
                PR_max = float(input("Enter maximum PR_max (in dBm): "))
                PR_step = float(input("Enter PR step (in dBm): "))
                config.varying_params['PRmax'] = np.arange(PR_min, PR_max + PR_step, PR_step)
            elif param == 'a':
                a_min = float(input("Enter minimum a (in dB): "))
                a_max = float(input("Enter maximum a (in dB): "))
                a_step = float(input("Enter a step (in dB): "))
                config.varying_params['a'] = np.arange(a_min, a_max + a_step, a_step)
            elif param == 'Pcn':
                pcn_min = float(input("Enter minimum Pcn (in dBm): "))
                pcn_max = float(input("Enter maximum Pcn (in dBm): "))
                pcn_step = float(input("Enter Pcn step (in dBm): "))
                config.varying_params['Pcn'] = np.arange(pcn_min, pcn_max + pcn_step, pcn_step)
        config.run_separately = input("Do you want to run the simulations for each parameter separately? (yes/no): ").strip().lower() == 'yes'
    return config

def simulation(sample_index, config, channel_samples_range, ris_samples_range, params_combo):
    print(f'\nStarting Simulation for sample_index: {sample_index}\n')
    channel_samples = channel_samples_range[sample_index] # G_B_lst, gE_hat_lst, gE_error_lst, H_lst, sigma_e_sq
    ris_samples = ris_samples_range[sample_index]
    
    index = 0
    # Set the parameters for this run based on params_combo
    for param, value in params_combo.items():
        unit, explanation = Utils.get_unit_and_explanation(param)
        print(f'\n{"*"*40}\nRunning Simulation! Varying {param}: {value} {unit}\n')
        print(f'{param}: {explanation}\n{"*"*40}\n')
        
        if param == 'Ptmax':
            config.Ptmax = config.dbm_to_watts(value)
        elif param == 'N':
            config.N = value
        elif param == 'PRmax':
            config.PRmax = config.dbm_to_watts(value)
        elif param == 'a':
            config.a = config.db_to_linear(value)
        elif param == 'Pcn':
            if config.ris_state == 'active':
                config.Pcn_a = value
            else:
                config.Pcn_p = value

    Pc_active = config.compute_static_power_consumption(config.N)
    Pc_passive = config.compute_static_power_consumption(config.N)
    Pc = Pc_active if config.ris_state == 'active' else Pc_passive

    G_B = channel_samples[config.N]['G_B']
    gE_hat = channel_samples[config.N]['gE_hat']
    gE_error = channel_samples[config.N]['gE_error']
    H = channel_samples[config.N]['H']
    sigma_e_sq = channel_samples[config.N]['sigma_e_sq']
    gE = gE_hat + gE_error
    gamma = ris_samples[config.N]['gamma']

    p_uniform = (config.Ptmax / config.K) * np.ones(config.K)
    R = Utils.compute_R(H, p_uniform, config.sigma_RIS_sq)
    assert np.all(np.linalg.eigvals(R) > 0), "Matrix R is not positive definite."

    if config.ris_state == 'active':
        PRmax = (config.a - 1) * np.real(np.trace(R)) if config.rf_state == 'RF-Gain' else config.PRmax
    else:
        PRmax = 0

    optimizer_psci = Optimizer(H, G_B, gE_hat, gE_error, config.sigma_sq, config.sigma_RIS_sq, sigma_e_sq, config.mu, Pc, PRmax, config.a, config.Ptmax, config.BW, scsi_bool=0)
    optimizer_scsi = Optimizer(H, G_B, gE_hat, gE_error, config.sigma_sq, config.sigma_RIS_sq, sigma_e_sq, config.mu, Pc, PRmax, config.a, config.Ptmax, config.BW, scsi_bool=1)

    rho_norm = np.sqrt((PRmax + np.real(np.trace(R))) / np.real(np.sum(gamma.conj().T @ R @ gamma))) if config.cons_state == 'global' else 1
    gamma_random = rho_norm * gamma

    p_init_pcsi = p_init_scsi = p_uniform
    gamma_init_pcsi = gamma_init_scsi = gamma_random
    
    if param != "N":
        if index != 0:
            p_init_pcsi = p_sol_pcsi
            gamma_init_pcsi = gamma_sol_pcsi
            p_init_scsi = p_sol_scsi
            gamma_init_scsi = gamma_sol_scsi

    sr_uniform_Bob_pcsi = Utils.SR_active_algo1(G_B, H, gamma_random, p_uniform, config.sigma_sq, config.sigma_RIS_sq, sigma_e_sq, scsi_bool=0, orig_bool=True, Rx="Bob")
    sr_uniform_Eve_pcsi = Utils.SR_active_algo1(gE, H, gamma_random, p_uniform, config.sigma_sq, config.sigma_RIS_sq, sigma_e_sq, scsi_bool=0, orig_bool=True, Rx="Eve")
    ssr_uniform_pcsi = max(sr_uniform_Bob_pcsi - sr_uniform_Eve_pcsi, 0)

    sr_uniform_Bob_scsi = Utils.SR_active_algo1(G_B, H, gamma_random, p_uniform, config.sigma_sq, config.sigma_RIS_sq, sigma_e_sq, scsi_bool=0, orig_bool=True, Rx="Bob")
    sr_uniform_Eve_scsi = Utils.SR_active_algo1(gE_hat, H, gamma_random, p_uniform, config.sigma_sq, config.sigma_RIS_sq, sigma_e_sq, scsi_bool=1, orig_bool=False, Rx="Eve")
    ssr_uniform_scsi = max(sr_uniform_Bob_scsi - sr_uniform_Eve_scsi, 0)

    gee_uniform_Bob_pcsi = config.BW * Utils.GEE_active_algo1(G_B, H, gamma_random, p_uniform, config.mu, Pc, config.sigma_sq, config.sigma_RIS_sq, sigma_e_sq, config.ris_state, scsi_bool=0, orig_bool=True, Rx="Bob")
    gee_uniform_Eve_pcsi = config.BW * Utils.GEE_active_algo1(gE, H, gamma_random, p_uniform, config.mu, Pc, config.sigma_sq, config.sigma_RIS_sq, sigma_e_sq, config.ris_state, scsi_bool=0, orig_bool=True, Rx="Eve")
    see_uniform_pcsi = max(gee_uniform_Bob_pcsi - gee_uniform_Eve_pcsi, 0)

    gee_uniform_Bob_scsi = config.BW * Utils.GEE_active_algo1(G_B, H, gamma_random, p_uniform, config.mu, Pc, config.sigma_sq, config.sigma_RIS_sq, sigma_e_sq, config.ris_state, scsi_bool=0, orig_bool=True, Rx="Bob")
    gee_uniform_Eve_scsi = config.BW * Utils.GEE_active_algo1(gE_hat, H, gamma_random, p_uniform, config.mu, Pc, config.sigma_sq, config.sigma_RIS_sq, sigma_e_sq, config.ris_state, scsi_bool=1, orig_bool=False, Rx="Eve")
    see_uniform_scsi = max(gee_uniform_Bob_scsi - gee_uniform_Eve_scsi, 0)

    print(f"\npCSI: Starting {config.ris_state.capitalize()} Secrecy {config.opt_state.upper()} optimization...\n")
    p_sol_pcsi, gamma_sol_pcsi, gamma_sol_Q_pcsi, ssr_sol_pcsi, ssr_sol_Q_pcsi, see_sol_pcsi, see_sol_Q_pcsi, iteration_altopt_pcsi, iteration_p_pcsi, iteration_gamma_pcsi, time_complexity_altopt_pcsi, time_complexity_p_pcsi, time_complexity_gamma_pcsi = optimizer_psci.altopt_algo1_mod(
        gamma_init_pcsi, p_init_pcsi, config.opt_bool[config.opt_state], config.rf_state, config.ris_state, config.cons_state, config.bits_range, config.quantization
    )

    print(f"\nsCSI: Starting {config.ris_state.capitalize()} Secrecy {config.opt_state.upper()} optimization...\n")
    p_sol_scsi, gamma_sol_scsi, gamma_sol_Q_scsi, ssr_sol_scsi, ssr_sol_Q_scsi, see_sol_scsi, see_sol_Q_scsi, iteration_altopt_scsi, iteration_p_scsi, iteration_gamma_scsi, time_complexity_altopt_scsi, time_complexity_p_scsi, time_complexity_gamma_scsi = optimizer_scsi.altopt_algo1_mod(
        gamma_init_scsi, p_init_scsi, config.opt_bool[config.opt_state], config.rf_state, config.ris_state, config.cons_state, config.bits_range, config.quantization
    )
    
    print(f'\n{"*"*40}\nStoring output values for (sample Index: {sample_index}, params_combo: {params_combo} {unit})\n{"*"*40}\n')
    output_data = Utils.save_output_data(
        config.NAMES, sample_index, params_combo, p_uniform, gamma_random, p_sol_pcsi, p_sol_scsi, gamma_sol_pcsi, gamma_sol_Q_pcsi, gamma_sol_scsi, gamma_sol_Q_scsi,
        sr_uniform_Bob_pcsi, sr_uniform_Bob_scsi, sr_uniform_Eve_pcsi, sr_uniform_Eve_scsi, ssr_uniform_pcsi, ssr_uniform_scsi,
        gee_uniform_Bob_pcsi, gee_uniform_Bob_scsi, gee_uniform_Eve_pcsi, gee_uniform_Eve_scsi, see_uniform_pcsi, see_uniform_scsi,
        ssr_sol_pcsi, ssr_sol_Q_pcsi, ssr_sol_scsi, ssr_sol_Q_scsi, see_sol_pcsi, see_sol_Q_pcsi, see_sol_scsi, see_sol_Q_scsi,
        iteration_altopt_pcsi, iteration_altopt_scsi, iteration_p_pcsi, iteration_p_scsi, iteration_gamma_pcsi, iteration_gamma_scsi,
        time_complexity_altopt_pcsi, time_complexity_altopt_scsi, time_complexity_p_pcsi, time_complexity_p_scsi, time_complexity_gamma_pcsi, time_complexity_gamma_scsi
    )
    
    index = index + 1
    
    return output_data

def simulation_task_wrapper(args):
    return simulation(*args)

def run_simulation(config):
    print("################################# STARTING SIMULATION: LET'S GO! #################################")
    num_processes = cpu_count()
    print(f"Number of CPU cores available: {num_processes}")

    N_range = config.varying_params.get('N', [config.N])
    channel_samples_range = Utils.monte_carlo_simulation(
        config.NUM_SAMPLES, config.FILENAME_CHANNEL, N_range, config.r_cell, config.h_BS, config.h_RIS, config.hmin_UE, config.hmax_UE, config.d0,  config.K, config.NR, config.f0, config.lambda_0, config.d, config.c, config.n, config.R_K, config.NEEV, config.vic_percent_eve, channel_model="rician", shadowing=False
    )

    ris_samples_range = Utils.generate_ris_coefficients(
        config.NUM_SAMPLES, config.FILENAME_RIS, N_range, channel_samples_range, (config.PTMIN / config.K) * np.ones(config.K),  config.sigma_sq, config.sigma_RIS_sq, scsi_bool=0, algo="algo1", state=config.ris_state
    )

    all_combinations = []
    if config.run_separately:
        for param, values in config.varying_params.items():
            for value in values:
                params_combo = {param: value}
                for sample_index in range(config.NUM_SAMPLES):
                    all_combinations.append((sample_index, config, channel_samples_range, ris_samples_range, params_combo))
    else:
        keys, values = zip(*config.varying_params.items())
        for combination in product(*values):
            params_combo = dict(zip(keys, combination))
            for sample_index in range(config.NUM_SAMPLES):
                all_combinations.append((sample_index, config, channel_samples_range, ris_samples_range, params_combo))

    with Pool(processes=num_processes) as pool:
        chunk_size = max(len(all_combinations) // num_processes, 1)
        output_results = pool.map(simulation_task_wrapper, all_combinations, chunksize=chunk_size)

    return output_results

if __name__ == '__main__':
    config = SystemConfig()
    
    introduce_problem()
    config = prompt_user_for_config(config)
    
    simulation_type, config = select_simulation_type(config)
    config = configure_simulation_params(simulation_type, config)
    
    # Run the simulation
    results = run_simulation(config)
    output_data = Utils.flatten_and_group_nested_list_of_dicts_ver2(results)
    np.savez_compressed(config.OUTPUT_FILE, output_data)
    print("******************************************** SIMULATION SUCCESSFULLY COMPLETED! CIAO 😇 ********************************************")








# ---------------------------------- OLD --------------------------------------------#
# import sys
# import os

# # Add the 'src' directory to the sys.path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# import numpy as np
# from multiprocessing import Pool, cpu_count
# from itertools import repeat, product
# from config import SystemConfig
# from utils import Utils
# from gamma_utils import GammaUtils
# from power_utils import PowerUtils
# from optimizers import Optimizer

# def introduce_problem():
#     print("""
#     ##############################################################
#     #    Resource Allocation and Energy Efficiency Management    #
#     #      in RIS-Aided Physical Layer Security Communication     #
#     ##############################################################
#     """)
#     print("""
#     This project addresses the problem of optimizing resource allocation and energy efficiency in RIS-aided uplink communication systems with a focus on physical layer security. We consider different simulation scenarios to evaluate the impact of various system parameters on the overall performance.
#     """)

# def prompt_user_for_config(config):
#     use_default = input("Do you want to use the default configuration? (yes/no): ").strip().lower()
    
#     if use_default == 'no':
#         config.rf_state = input("Enter RF state ('RF-Gain' or 'RF-Power'): ").strip().lower()
#         config.ris_state = input("Enter RIS state ('active' or 'passive'): ").strip().lower()
#         config.cons_state = input("Enter constraint state ('global' or 'local'): ").strip().lower()
#         config.opt_state = input("Enter optimization state ('sr' or 'ee'): ").strip().lower()
#         config.quantization = True if input("Do you want to enable quantization? (yes/no): ").strip().lower() == 'yes' else False
#         config.bits_range = eval(input("Enter bits range as a list of tuples, e.g., [(1,1), (2,2), (3,3), (4,4)]: "))
        
#         config.K = int(input("Enter number of UEs: "))
#         config.NR = [int(x) for x in input("Enter number of receive antennas at the BS and Eve (comma-separated): ").strip().split(',')]
#         config.BW = float(input("Enter system bandwidth in Hz: "))
#         config.f0 = float(input("Enter operating frequency in Hz: "))
#         config.r_cell = float(input("Enter cell radius in meters: "))
#         config.h_BS = float(input("Enter height of the BS in meters: "))
#         config.h_RIS = float(input("Enter height of the RIS in meters: "))
#         config.hmin_UE = float(input("Enter minimum height for UEs in meters: "))
#         config.hmax_UE = float(input("Enter maximum height for UEs in meters: "))
#         config.NUM_SAMPLES = int(input("Enter number of samples: "))
        
#         fixed_N = int(input("Enter fixed value for N: "))
#         fixed_Ptmax = float(input("Enter fixed value for Ptmax (dBm): "))
#         fixed_PRmax = float(input("Enter fixed value for PRmax (dBm): "))
#         fixed_a = float(input("Enter fixed value for a (dB): "))
#         fixed_Pcn_p = float(input("Enter fixed value for Pcn_p (dBm): "))
#         fixed_Pcn_a = float(input("Enter fixed value for Pcn_a (dBm): "))
    
#         config.update_custom_params(N=fixed_N, Ptmax=fixed_Ptmax, PRmax=fixed_PRmax, a=fixed_a, Pcn_p=fixed_Pcn_p, Pcn_a=fixed_Pcn_a)

#     return config

# def select_simulation_type(config):
#     print("""
#     Select the type of simulation to run:
#     1. SSR, SEE Vs Ptmax
#     2. SSR, SEE Vs N
#     3. SSR, SEE Vs PRmax
#     4. SSR, SEE Vs a
#     5. SSR, SEE Vs Pcn
#     6. Customize (Select multiple parameters to vary)
#     """)
#     simulation_type = int(input("Enter the number corresponding to the simulation type: "))
#     config.update_output_file(simulation_type)
#     return simulation_type, config

# def configure_simulation_params(simulation_type, config):
#     if simulation_type == 1:
#         ptmax_min = float(input("Enter minimum Ptmax (in dBm): "))
#         ptmax_max = float(input("Enter maximum Ptmax (in dBm): "))
#         ptmax_step = float(input("Enter Ptmax step (in dBm): "))
#         config.varying_params = {'Ptmax': np.arange(ptmax_min, ptmax_max + ptmax_step, ptmax_step)}
#     elif simulation_type == 2:
#         n_min = int(input("Enter minimum N (number of RIS elements): "))
#         n_max = int(input("Enter maximum N (number of RIS elements): "))
#         n_step = int(input("Enter N step: "))
#         config.varying_params = {'N': range(n_min, n_max + n_step, n_step)}
#     elif simulation_type == 3:
#         PR_min = float(input("Enter minimum PR_min (in dBm): "))
#         PR_max = float(input("Enter maximum PR_max (in dBm): "))
#         PR_step = float(input("Enter a step (in dBm): "))
#         config.varying_params = {'PRmax': np.arange(PR_min, PR_max + PR_step, PR_step)}
#     elif simulation_type == 4:
#         a_min = float(input("Enter minimum a (in dB): "))
#         a_max = float(input("Enter maximum a (in dB): "))
#         a_step = float(input("Enter a step (in dB): "))
#         config.varying_params = {'a': np.arange(a_min, a_max + a_step, a_step)}
#     elif simulation_type == 5:
#         pcn_min = float(input("Enter minimum Pcn (in dBm): "))
#         pcn_max = float(input("Enter maximum Pcn (in dBm): "))
#         pcn_step = float(input("Enter Pcn step (in dBm): "))
#         config.varying_params = {'Pcn': np.arange(pcn_min, pcn_max + pcn_step, pcn_step)}
#     elif simulation_type == 6:
#         config.varying_params = {}
#         print("Enter the parameters you want to vary (comma-separated): Ptmax, N, a, Pcn")
#         params = input().strip().split(',')
#         for param in params:
#             param = param.strip()
#             if param == 'Ptmax':
#                 ptmax_min = float(input("Enter minimum Ptmax (in dBm): "))
#                 ptmax_max = float(input("Enter maximum Ptmax (in dBm): "))
#                 ptmax_step = float(input("Enter Ptmax step (in dBm): "))
#                 config.varying_params['Ptmax'] = np.arange(ptmax_min, ptmax_max + ptmax_step, ptmax_step)
#             elif param == 'N':
#                 n_min = int(input("Enter minimum N (number of RIS elements): "))
#                 n_max = int(input("Enter maximum N (number of RIS elements): "))
#                 n_step = int(input("Enter N step: "))
#                 config.varying_params['N'] = range(n_min, n_max + n_step, n_step)
#             elif param == 'PRmax':
#                 PR_min = float(input("Enter minimum PR_min (in dBm): "))
#                 PR_max = float(input("Enter maximum PR_max (in dBm): "))
#                 PR_step = float(input("Enter PR step (in dBm): "))
#                 config.varying_params['PRmax'] = np.arange(PR_min, PR_max + PR_step, PR_step)
#             elif param == 'a':
#                 a_min = float(input("Enter minimum a (in dB): "))
#                 a_max = float(input("Enter maximum a (in dB): "))
#                 a_step = float(input("Enter a step (in dB): "))
#                 config.varying_params['a'] = np.arange(a_min, a_max + a_step, a_step)
#             elif param == 'Pcn':
#                 pcn_min = float(input("Enter minimum Pcn (in dBm): "))
#                 pcn_max = float(input("Enter maximum Pcn (in dBm): "))
#                 pcn_step = float(input("Enter Pcn step (in dBm): "))
#                 config.varying_params['Pcn'] = np.arange(pcn_min, pcn_max + pcn_step, pcn_step)
#         config.run_separately = input("Do you want to run the simulations for each parameter separately? (yes/no): ").strip().lower() == 'yes'
#     return config

# def simulation(sample_index, config, channel_samples_range, ris_samples_range, params_combo):
#     print(f'\nStarting Simulation for sample_index: {sample_index}\n')
#     channel_samples = channel_samples_range[sample_index] # G_B_lst, gE_hat_lst, gE_error_lst, H_lst, sigma_e_sq
#     ris_samples = ris_samples_range[sample_index]
#     index = 0
#     # Set the parameters for this run based on params_combo
#     for param, value in params_combo.items():
        
#         unit, explanation = Utils.get_unit_and_explanation(param)
#         print(f'\n{"*"*40}\nRunning Simulation! Varying {param}: {value} {unit}\n')
#         print(f'{param}: {explanation}\n{"*"*40}\n')
        
#         if param == 'Ptmax':
#             config.Ptmax = config.dbm_to_watts(value)
#         elif param == 'N':
#             config.N = value
#         elif param == 'PRmax':
#             config.PRmax = config.dbm_to_watts(value)
#         elif param == 'a':
#             config.a = config.db_to_linear(value)
#         elif param == 'Pcn':
#             if config.ris_state == 'active':
#                 config.Pcn_a = value
#             else:
#                 config.Pcn_p = value

#     Pc_active = config.compute_static_power_consumption(config.N)
#     Pc_passive = config.compute_static_power_consumption(config.N)
#     Pc = Pc_active if config.ris_state == 'active' else Pc_passive

#     G_B = channel_samples[config.N]['G_B']
#     gE_hat = channel_samples[config.N]['gE_hat']
#     gE_error = channel_samples[config.N]['gE_error']
#     H = channel_samples[config.N]['H']
#     sigma_e_sq = channel_samples[config.N]['sigma_e_sq']
#     gE = gE_hat + gE_error
#     gamma = ris_samples[config.N]['gamma']

#     p_uniform = (config.Ptmax / config.K) * np.ones(config.K)
#     R = Utils.compute_R(H, p_uniform, config.sigma_RIS_sq)
#     assert np.all(np.linalg.eigvals(R) > 0), "Matrix R is not positive definite."

#     if config.ris_state == 'active':
#         PRmax = (config.a - 1) * np.real(np.trace(R)) if config.rf_state == 'RF-Gain' else config.PRmax
#     else:
#         PRmax = 0

#     optimizer_psci = Optimizer(H, G_B, gE_hat, gE_error, config.sigma_RIS_sq, config.sigma_RIS_sq, sigma_e_sq, config.mu, Pc, PRmax, config.a, config.Ptmax, config.BW, scsi_bool=0)
#     optimizer_scsi = Optimizer(H, G_B, gE_hat, gE_error, config.sigma_RIS_sq, config.sigma_RIS_sq, sigma_e_sq, config.mu, Pc, PRmax, config.a, config.Ptmax, config.BW, scsi_bool=1)

#     rho_norm = np.sqrt((PRmax + np.real(np.trace(R))) / np.real(np.sum(gamma.conj().T @ R @ gamma))) if config.cons_state == 'global' else 1
#     gamma_random = rho_norm * gamma

#     p_init_pcsi = p_init_scsi = p_uniform
#     gamma_init_pcsi = gamma_init_scsi = gamma_random
    
#     if param != "N":
#         if index == 0:
#             p_init_pcsi = p_init_scsi = p_uniform
#             gamma_init_pcsi = gamma_init_scsi = gamma_random
#         else:
#             p_init_pcsi = p_sol_pcsi
#             gamma_init_pcsi = gamma_sol_pcsi
#             p_init_scsi = p_sol_scsi
#             gamma_init_scsi = gamma_sol_scsi

#     sr_uniform_Bob_pcsi = Utils.SR_active_algo1(G_B, H, gamma_random, p_uniform, config.sigma_RIS_sq, config.sigma_RIS_sq, sigma_e_sq, scsi_bool=0, orig_bool=True, Rx="Bob")
#     sr_uniform_Eve_pcsi = Utils.SR_active_algo1(gE, H, gamma_random, p_uniform, config.sigma_RIS_sq, config.sigma_RIS_sq, sigma_e_sq, scsi_bool=0, orig_bool=True, Rx="Eve")
#     ssr_uniform_pcsi = max(sr_uniform_Bob_pcsi - sr_uniform_Eve_pcsi, 0)

#     sr_uniform_Bob_scsi = Utils.SR_active_algo1(G_B, H, gamma_random, p_uniform, config.sigma_RIS_sq, config.sigma_RIS_sq, sigma_e_sq, scsi_bool=0, orig_bool=True, Rx="Bob")
#     sr_uniform_Eve_scsi = Utils.SR_active_algo1(gE_hat, H, gamma_random, p_uniform, config.sigma_RIS_sq, config.sigma_RIS_sq, sigma_e_sq, scsi_bool=1, orig_bool=False, Rx="Eve")
#     ssr_uniform_scsi = max(sr_uniform_Bob_scsi - sr_uniform_Eve_scsi, 0)

#     gee_uniform_Bob_pcsi = config.BW * Utils.GEE_active_algo1(G_B, H, gamma_random, p_uniform, config.mu, Pc, config.sigma_RIS_sq, config.sigma_RIS_sq, sigma_e_sq, config.ris_state, scsi_bool=0, orig_bool=True, Rx="Bob")
#     gee_uniform_Eve_pcsi = config.BW * Utils.GEE_active_algo1(gE, H, gamma_random, p_uniform, config.mu, Pc, config.sigma_RIS_sq, config.sigma_RIS_sq, sigma_e_sq, config.ris_state, scsi_bool=0, orig_bool=True, Rx="Eve")
#     see_uniform_pcsi = max(gee_uniform_Bob_pcsi - gee_uniform_Eve_pcsi, 0)

#     gee_uniform_Bob_scsi = config.BW * Utils.GEE_active_algo1(G_B, H, gamma_random, p_uniform, config.mu, Pc, config.sigma_RIS_sq, config.sigma_RIS_sq, sigma_e_sq, config.ris_state, scsi_bool=0, orig_bool=True, Rx="Bob")
#     gee_uniform_Eve_scsi = config.BW * Utils.GEE_active_algo1(gE_hat, H, gamma_random, p_uniform, config.mu, Pc, config.sigma_RIS_sq, config.sigma_RIS_sq, sigma_e_sq, config.ris_state, scsi_bool=1, orig_bool=False, Rx="Eve")
#     see_uniform_scsi = max(gee_uniform_Bob_scsi - gee_uniform_Eve_scsi, 0)

#     print(f"\npCSI: Starting {config.ris_state.capitalize()} Secrecy {config.opt_state.upper()} optimization...\n")
#     p_sol_pcsi, gamma_sol_pcsi, gamma_sol_Q_pcsi, ssr_sol_pcsi, ssr_sol_Q_pcsi, see_sol_pcsi, see_sol_Q_pcsi, iteration_altopt_pcsi, iteration_p_pcsi, iteration_gamma_pcsi, time_complexity_altopt_pcsi, time_complexity_p_pcsi, time_complexity_gamma_pcsi = optimizer_psci.altopt_algo1_mod(
#         gamma_init_pcsi, p_init_pcsi, config.opt_bool[config.opt_state], config.rf_state, config.ris_state, config.cons_state, config.bits_range, config.quantization
#     )

#     print(f"\nsCSI: Starting {config.ris_state.capitalize()} Secrecy {config.opt_state.upper()} optimization...\n")
#     p_sol_scsi, gamma_sol_scsi, gamma_sol_Q_scsi, ssr_sol_scsi, ssr_sol_Q_scsi, see_sol_scsi, see_sol_Q_scsi, iteration_altopt_scsi, iteration_p_scsi, iteration_gamma_scsi, time_complexity_altopt_scsi, time_complexity_p_scsi, time_complexity_gamma_scsi = optimizer_scsi.altopt_algo1_mod(
#         gamma_init_scsi, p_init_scsi, config.opt_bool[config.opt_state], config.rf_state, config.ris_state, config.cons_state, config.bits_range, config.quantization
#     )
    
#     print(f'\n{"*"*40}\nStoring output values for (sample Index: {sample_index}, {param}_Index: {index}, {param}_value: {value} {unit})\n{"*"*40}\n')
#     output_data = Utils.save_output_data(
#         config.NAMES, sample_index, params_combo, p_uniform, gamma_random, p_sol_pcsi, p_sol_scsi, gamma_sol_pcsi, gamma_sol_Q_pcsi, gamma_sol_scsi, gamma_sol_Q_scsi,
#         sr_uniform_Bob_pcsi, sr_uniform_Bob_scsi, sr_uniform_Eve_pcsi, sr_uniform_Eve_scsi, ssr_uniform_pcsi, ssr_uniform_scsi,
#         gee_uniform_Bob_pcsi, gee_uniform_Bob_scsi, gee_uniform_Eve_pcsi, gee_uniform_Eve_scsi, see_uniform_pcsi, see_uniform_scsi,
#         ssr_sol_pcsi, ssr_sol_Q_pcsi, ssr_sol_scsi, ssr_sol_Q_scsi, see_sol_pcsi, see_sol_Q_pcsi, see_sol_scsi, see_sol_Q_scsi,
#         iteration_altopt_pcsi, iteration_altopt_scsi, iteration_p_pcsi, iteration_p_scsi, iteration_gamma_pcsi, iteration_gamma_scsi,
#         time_complexity_altopt_pcsi, time_complexity_altopt_scsi, time_complexity_p_pcsi, time_complexity_p_scsi, time_complexity_gamma_pcsi, time_complexity_gamma_scsi
#     )
    
#     index = index + 1
#     return output_data

# def simulation_task_wrapper(args):
#     return simulation(*args)

# def run_simulation(config):
#     print("################################# STARTING SIMULATION: LETS GO! #################################")
#     num_processes = cpu_count()
#     print(f"Number of CPU cores available: {num_processes}")

#     N_range = config.varying_params.get('N', [config.N])
#     channel_samples_range = Utils.monte_carlo_simulation(
#         config.NUM_SAMPLES, config.FILENAME_CHANNEL, N_range, config.r_cell, config.h_BS, config.h_RIS, config.hmin_UE, config.hmax_UE, config.d0,  config.K, config.NR, config.f0, config.lambda_0, config.d, config.c, config.n, config.R_K, config.NEEV, config.vic_percent_eve, channel_model="rician", shadowing=False
#     )

#     ris_samples_range = Utils.generate_ris_coefficients(
#         config.NUM_SAMPLES, config.FILENAME_RIS, N_range, channel_samples_range, (config.PTMIN / config.K) * np.ones(config.K), config.sigma_RIS_sq, config.sigma_RIS_sq, scsi_bool=0, algo="algo1", state='active'
#     )

#     all_combinations = []
#     if config.run_separately:
#         for param, values in config.varying_params.items():
#             for value in values:
#                 params_combo = {param: value}
#                 for sample_index in range(config.NUM_SAMPLES):
#                     all_combinations.append((sample_index, config, channel_samples_range, ris_samples_range, params_combo))
#     else:
#         keys, values = zip(*config.varying_params.items())
#         for combination in product(*values):
#             params_combo = dict(zip(keys, combination))
#             for sample_index in range(config.NUM_SAMPLES):
#                 all_combinations.append((sample_index, config, channel_samples_range, ris_samples_range, params_combo))

#     with Pool(processes=num_processes) as pool:
#         output_results = pool.map(simulation_task_wrapper, all_combinations)

#     return output_results

# if __name__ == '__main__':
#     config = SystemConfig()
    
#     introduce_problem()
#     config = prompt_user_for_config(config)
    
#     simulation_type, config = select_simulation_type(config)
#     config = configure_simulation_params(simulation_type, config)
    
#     # Run the simulation
#     results = run_simulation(config)
#     output_data = Utils.flatten_and_group_nested_list_of_dicts(results)
#     np.savez_compressed(config.OUTPUT_FILE, output_data)
#     print("******************************************** SIMULATION SUCCESSFULLY COMPLETED! CIAO 😇 ********************************************")