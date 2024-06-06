import unittest
import numpy as np
from power_utils import PowerUtils

class TestPowerUtils(unittest.TestCase):
    
    def setUp(self):
        self.power_utils = PowerUtils()

    def test_compute_Pc_eq_p(self):
        gamma = np.random.randn(4)
        Pc = 1
        sigma_RIS_sq = 1e-3
        Pc_eq = self.power_utils.compute_Pc_eq_p(gamma, Pc, sigma_RIS_sq)
        self.assertIsInstance(Pc_eq, float)

    def test_compute_mu_eq_p(self):
        H = np.random.randn(4, 4)
        gamma = np.random.randn(4)
        mu = 1
        mu_eq = self.power_utils.compute_mu_eq_p(H, gamma, mu)
        self.assertEqual(mu_eq.shape, (4,))

if __name__ == '__main__':
    unittest.main()
