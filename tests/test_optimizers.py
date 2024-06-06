import unittest
import numpy as np
from optimizers import Optimizer

class TestOptimizer(unittest.TestCase):
    
    def setUp(self):
        self.optimizer = Optimizer()

    def test_gamma_cvxopt_algo1_mod(self):
        G_B = np.random.randn(4, 4)
        gE = np.random.randn(4, 4)
        H = np.random.randn(4, 4)
        gamma = np.random.randn(4)
        p = np.random.randn(4)
        BW, mu, a, Pc = 20e6, 1, 2, 1
        sigma_sq, sigma_RIS_sq, sigma_g_sq = 1e-3, 1e-3, 1e-3
        opt_bool, ris_state, cons_state, scsi_bool = 1, 'active', 'global', 1

        result = self.optimizer.gamma_cvxopt_algo1_mod(G_B, gE, H, gamma, p, BW, mu, a, Pc, sigma_sq, sigma_RIS_sq, sigma_g_sq, opt_bool, ris_state, cons_state, scsi_bool)
        self.assertEqual(len(result), 5)

if __name__ == '__main__':
    unittest.main()
