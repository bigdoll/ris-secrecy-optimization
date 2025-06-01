# import numpy as np

# def average_values_by_key_equal_length(flattened_dict, num_samples):
#         """
#         Average values by key in a flattened dictionary.

#         Parameters:
#         - flattened_dict (dict): Flattened dictionary.
#         - num_samples (int): Number of samples.

#         Returns:
#         - dict: Dictionary with averaged values.
#         """
#         keys_to_average = {
#             "sr_uniform_Bob_pcsi", "sr_uniform_Bob_scsi", "sr_uniform_Eve_pcsi",
#             "sr_uniform_Eve_scsi", "ssr_uniform_pcsi", "ssr_uniform_scsi", "gee_uniform_Bob_pcsi", "gee_uniform_Bob_scsi",
#             "gee_uniform_Eve_pcsi", "gee_uniform_Eve_scsi", "see_uniform_pcsi", "see_uniform_scsi", "ssr_sol_pcsi", "ssr_sol_Q_pcsi",
#             "ssr_sol_scsi", "ssr_sol_Q_scsi", "see_sol_pcsi", "see_sol_Q_pcsi", "see_sol_scsi", "see_sol_Q_scsi", "iteration_altopt_pcsi", "iteration_altopt_scsi",
#             "iteration_p_pcsi", "iteration_p_scsi", "iteration_gamma_pcsi", "iteration_gamma_scsi",
#             "time_complexity_altopt_pcsi", "time_complexity_altopt_scsi", "time_complexity_p_pcsi",
#             "time_complexity_p_scsi", "time_complexity_gamma_pcsi", "time_complexity_gamma_scsi"
#         }

#         avg_dict = {}
#         grouped_dict = {}
#         for i in range(num_samples):
#             for key, value_lists in flattened_dict[i].items():
#                if key in {"ssr_sol_Q_pcsi", "ssr_sol_Q_scsi", "see_sol_Q_pcsi", "see_sol_Q_scsi"}:
#                         # Special case: list of dictionaries     
#                     for ii, sample_dicts in enumerate(value_lists):
#                           for k, v in sample_dicts.items():
#                               if k not in grouped_dict.keys():
#                                   grouped_dict[key][k] = [[] for _ in range(num_samples)]
#                               grouped_dict[key][k][i].append(v)
                                  
#                else:
#                   avg_dict[key] = value_lists  # Preserve original values for keys not in keys_to_average
        
#         avg_dict_i = {}
#         for k, values_list in grouped_dict.items():
#               avg_list_i = [sum(values) / num_samples for values in zip(*values_list)]
#               avg_dict_i[k] = avg_list_i
                    
#         avg_dict[key] = avg_dict_i  
                

#         return avg_dict

# results_ee = np.load('data/outputs/output_results_algo1_ee_active_2s_100ris_10.0dB_5dBm_Ptmax.npz', allow_pickle=True)['arr_0'].item()

# test = average_values_by_key_equal_length(results_ee, 2)


# from collections import defaultdict
# from typing import List, Dict, Any

# def average_lists(lists: List[List[float]]) -> List[float]:
#     """
#     Averages the values in a list of lists element-wise.
#     """
#     if not lists:
#         return []
    
#     # Debug statement to check the input
#     print(f"Averaging lists: {lists}")

#     averaged = [sum(values) / len(values) for values in zip(*lists)]
#     return averaged

# def average_nested_dicts(dicts: List[Dict]) -> Dict:
#     """
#     Averages the values in a list of nested dictionaries element-wise.
#     """
#     if not dicts:
#         return {}
    
#     keys = dicts[0].keys()

#     # Debug statement to check the input
#     print(f"Averaging nested dictionaries for keys: {keys}")

#     averaged = {key: average_lists([d[key] for d in dicts]) for key in keys}
#     return averaged

# def average_results(results: Dict[int, Dict[str, Any]], keys_to_average: set) -> Dict[str, Any]:
#     """
#     Averages the key-value list across the sample indexes element-wise,
#     while maintaining the structure and leaving the parts not inside the
#     selected key names unchanged.
#     """
#     averaged_results = defaultdict(dict)
    
#     for key in results[0].keys():
#         if key in keys_to_average:
#             # Gather the lists to average
#             lists_to_average = [results[idx][key] for idx in results]
            
#             # Debug statement to check the input
#             print(f"Processing key: {key}, lists to average: {lists_to_average}")

#             # Determine if the elements are lists of numbers or lists of dictionaries
#             if all(isinstance(item, (int, float)) for sublist in lists_to_average for item in sublist):
#                 averaged_results[key] = average_lists(lists_to_average)
#             elif all(isinstance(item, dict) for sublist in lists_to_average for item in sublist):
#                 # Need to process each sub-list separately
#                 averaged_results[key] = [average_nested_dicts([d[i] for d in lists_to_average]) for i in range(len(lists_to_average[0]))]
#         else:
#             # Copy the values directly (unchanged)
#             for idx in results:
#                 averaged_results[idx][key] = results[idx][key]
    
#     return averaged_results

# # Example usage
# results_ee = {
#     0: {'p_uniform': [1, 2, 3], 'gamma_random': [4, 5, 6], 'sr_uniform_Bob_pcsi': [0.1, 0.2, 0.3], 'ssr_sol_Q_pcsi': [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]},
#     1: {'p_uniform': [2, 3, 4], 'gamma_random': [5, 6, 7], 'sr_uniform_Bob_pcsi': [0.2, 0.3, 0.4], 'ssr_sol_Q_pcsi': [{'a': 2, 'b': 3}, {'a': 4, 'b': 5}]}
# }

# keys_to_average = {
#     "sr_uniform_Bob_pcsi", "sr_uniform_Bob_scsi", "sr_uniform_Eve_pcsi",
#     "sr_uniform_Eve_scsi", "ssr_uniform_pcsi", "ssr_uniform_scsi", "gee_uniform_Bob_pcsi", "gee_uniform_Bob_scsi",
#     "gee_uniform_Eve_pcsi", "gee_uniform_Eve_scsi", "see_uniform_pcsi", "see_uniform_scsi", "ssr_sol_pcsi", "ssr_sol_Q_pcsi",
#     "ssr_sol_scsi", "ssr_sol_Q_scsi", "see_sol_pcsi", "see_sol_Q_pcsi", "see_sol_scsi", "see_sol_Q_scsi", "iteration_altopt_pcsi", "iteration_altopt_scsi",
#     "iteration_p_pcsi", "iteration_p_scsi", "iteration_gamma_pcsi", "iteration_gamma_scsi",
#     "time_complexity_altopt_pcsi", "time_complexity_altopt_scsi", "time_complexity_p_pcsi",
#     "time_complexity_p_scsi", "time_complexity_gamma_pcsi", "time_complexity_gamma_scsi"
# }

# averaged_results = average_results(results_ee, keys_to_average)
# print(averaged_results)


from collections import defaultdict
from typing import List, Dict, Any
import numpy as np

# def average_lists(lists: List[List[float]]) -> List[float]:
#     """
#     Averages the values in a list of lists element-wise.
#     """
#     if not lists:
#         return []
    
#     # Debug statement to check the input
#     print(f"Averaging lists: {lists}")

#     averaged = [sum(values) / len(values) for values in zip(*lists)]
#     return averaged

# def average_nested_dicts(dicts: List[Dict]) -> Dict:
#     """
#     Averages the values in a list of nested dictionaries element-wise.
#     """
#     if not dicts:
#         return {}
    
#     keys = dicts[0].keys()

#     # Debug statement to check the input
#     print(f"Averaging nested dictionaries for keys: {keys}")

#     averaged = {}
#     for key in keys:
#         values = [d[key] for d in dicts]
#         # Ensure the values are lists before averaging
#         if all(isinstance(v, (int, float)) for v in values):
#             averaged[key] = sum(values) / len(values)
#         else:
#             averaged[key] = average_lists(values)
#     return averaged

# def average_results(results: Dict[int, Dict[str, Any]], keys_to_average: set) -> Dict[str, Any]:
#     """
#     Averages the key-value list across the sample indexes element-wise,
#     while maintaining the structure and leaving the parts not inside the
#     selected key names unchanged.
#     """
#     averaged_results = defaultdict(dict)
    
#     for key in results[0].keys():
#         if key in keys_to_average:
#             # Gather the lists to average
#             lists_to_average = [results[idx][key] for idx in results]
            
#             # Debug statement to check the input
#             print(f"Processing key: {key}, lists to average: {lists_to_average}")

#             # Determine if the elements are lists of numbers or lists of dictionaries
#             if all(isinstance(item, (int, float)) for sublist in lists_to_average for item in sublist):
#                 averaged_results[key] = average_lists(lists_to_average)
#             elif all(isinstance(item, dict) for sublist in lists_to_average for item in sublist):
#                 # Need to process each sub-list separately
#                 num_elements = len(lists_to_average[0])
#                 averaged_results[key] = [average_nested_dicts([d[i] for d in lists_to_average]) for i in range(num_elements)]
#             else:
#                 print(f"Unhandled data structure for key: {key}")
#         else:
#             # Copy the values directly (unchanged)
#             for idx in results:
#                 averaged_results[idx][key] = results[idx][key]
    
#     return averaged_results



def average_lists(lists: List[List[float]]) -> List[float]:
    """
    Averages the values in a list of lists element-wise.
    """
    if not lists:
        return []
    
    # Debug statement to check the input
    print(f"Averaging lists: {lists}")

    averaged = [sum(values) / len(values) for values in zip(*lists)]
    return averaged

def average_nested_dicts(dicts: List[Dict]) -> Dict:
    """
    Averages the values in a list of nested dictionaries element-wise.
    """
    if not dicts:
        return {}
    
    keys = dicts[0].keys()

    # Debug statement to check the input
    print(f"Averaging nested dictionaries for keys: {keys}")

    averaged = {}
    for key in keys:
        values = [d[key] for d in dicts]
        # Ensure the values are lists before averaging
        if all(isinstance(v, (int, float)) for v in values):
            averaged[key] = sum(values) / len(values)
        else:
            averaged[key] = average_lists(values)
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
            print(f"Processing key: {key}, lists to average: {lists_to_average}")

            # Determine if the elements are lists of numbers or lists of dictionaries
            if all(isinstance(item, (int, float)) for sublist in lists_to_average for item in sublist):
                averaged_results[key] = average_lists(lists_to_average)
            elif all(isinstance(item, dict) for sublist in lists_to_average for item in sublist):
                # Group values by their keys across all sub-lists
                grouped_values = defaultdict(lambda: [[] for _ in range(len(lists_to_average))])
                for idx, sublist in enumerate(lists_to_average):
                    for d in sublist:
                        for k, v in d.items():
                            grouped_values[k][idx].append(v)
                
                # Convert grouped_values to lists of lists for averaging
                averaged_results[key] = {k: average_lists(v) for k, v in grouped_values.items()}
            else:
                print(f"Unhandled data structure for key: {key}")
        else:
            # Copy the values directly (unchanged)
            for idx in results:
                averaged_results[idx][key] = results[idx][key]
    
    return averaged_results

# Example usage
# results_ee = {
#     0: {'p_uniform': [1, 2, 3], 'gamma_random': [4, 5, 6], 'sr_uniform_Bob_pcsi': [0.1, 0.2, 0.3], 'ssr_sol_Q_pcsi': [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]},
#     1: {'p_uniform': [2, 3, 4], 'gamma_random': [5, 6, 7], 'sr_uniform_Bob_pcsi': [0.2, 0.3, 0.4], 'ssr_sol_Q_pcsi': [{'a': 2, 'b': 3}, {'a': 4, 'b': 5}]}
# }

results_ee = np.load('data/outputs/output_results_algo1_ee_active_2s_100ris_10.0dB_5dBm_Ptmax.npz', allow_pickle=True)['arr_0'].item()

keys_to_average = {
    "sr_uniform_Bob_pcsi", "sr_uniform_Bob_scsi", "sr_uniform_Eve_pcsi",
    "sr_uniform_Eve_scsi", "ssr_uniform_pcsi", "ssr_uniform_scsi", "gee_uniform_Bob_pcsi", "gee_uniform_Bob_scsi",
    "gee_uniform_Eve_pcsi", "gee_uniform_Eve_scsi", "see_uniform_pcsi", "see_uniform_scsi", "ssr_sol_pcsi", "ssr_sol_Q_pcsi",
    "ssr_sol_scsi", "ssr_sol_Q_scsi", "see_sol_pcsi", "see_sol_Q_pcsi", "see_sol_scsi", "see_sol_Q_scsi", "iteration_altopt_pcsi", "iteration_altopt_scsi",
    "iteration_p_pcsi", "iteration_p_scsi", "iteration_gamma_pcsi", "iteration_gamma_scsi",
    "time_complexity_altopt_pcsi", "time_complexity_altopt_scsi", "time_complexity_p_pcsi",
    "time_complexity_p_scsi", "time_complexity_gamma_pcsi", "time_complexity_gamma_scsi"
}

averaged_results = average_results(results_ee, keys_to_average)
print(averaged_results)



