import unittest
import numpy as np
from utils import Utils

class TestUtils(unittest.TestCase):
    
    def setUp(self):
        self.utils = Utils()

    def test_monte_carlo_simulation(self):
        # Mocked parameters for simulation
        NUM_SAMPLES = 5
        FILENAME = "mock_channel_samples.npy"
        r_cell, h_BS, h_RIS, hmin_UE, hmax_UE = 50, 10, 10, 1.5, 2.5
        d0, N_RANGE, K, NR, f0, lambda_0, d, c = 35, range(10, 60, 10), 4, [4, 1], 3.5e9, 0.085714, 0.042857, 3e8
        n, R_K, sigma_g_sq = [4, 2, 4], [2, 4, 2], 1e-3
        algo = "algo2"

        channel_samples = self.utils.monte_carlo_simulation(
            NUM_SAMPLES, FILENAME, r_cell, h_BS, h_RIS, hmin_UE, hmax_UE, d0, N_RANGE, K, NR, f0, lambda_0, d, c, n, R_K, sigma_g_sq, algo
        )
        self.assertEqual(len(channel_samples), NUM_SAMPLES)

    def test_generate_ris_coefficients(self):
        NUM_SAMPLES = 5
        FILENAME = "mock_ris_coefficients.npy"
        N_RANGE = range(10, 60, 10)
        channel_samples = np.random.randn(NUM_SAMPLES, 4)
        P_TEMP = np.ones(4)
        sigma_sq, sigma_RIS_sq, sigma_g_sq = 1e-3, 1e-3, 1e-3
        scsi_bool, algo, state = 0, "algo1", 'active'

        ris_samples = self.utils.generate_ris_coefficients(
            NUM_SAMPLES, FILENAME, N_RANGE, channel_samples, P_TEMP, sigma_sq, sigma_RIS_sq, sigma_g_sq, scsi_bool, algo, state
        )
        self.assertEqual(len(ris_samples), NUM_SAMPLES)

    def test_flatten_and_group_nested_list_of_dicts(self):
        nested_list = [[{'a': 1, 'b': 2}], [{'a': 3, 'b': 4}]]
        flattened = self.utils.flatten_and_group_nested_list_of_dicts(nested_list)
        self.assertEqual(len(flattened), 2)

if __name__ == '__main__':
    unittest.main()
