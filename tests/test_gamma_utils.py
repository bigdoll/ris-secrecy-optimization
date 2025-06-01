import unittest
import numpy as np
from gamma_utils import GammaUtils

class TestGammaUtils(unittest.TestCase):
    
    def setUp(self):
        H = np.random.randn(4, 4)
        G_B = np.random.randn(4, 4)
        G_E = np.random.randn(4, 4)
        sigma_sq, sigma_RIS_sq, sigma_g_sq = 1e-3, 1e-3, 1e-3
        mu, Pc, scsi_bool = 1, 1, 1
        self.gamma_utils = GammaUtils(H, G_B, G_E, sigma_sq, sigma_RIS_sq, sigma_g_sq, mu, Pc, scsi_bool)

    def test_compute_R(self):
        p = np.random.randn(4)
        R = self.gamma_utils.compute_R(p)
        self.assertEqual(R.shape, (4, 4))

    def test_parameters_active_Bob(self):
        C = np.random.randn(4, 4)
        gamma_bar = np.random.randn(4)
        p = np.random.randn(4)
        params = self.gamma_utils.parameters_active_Bob(C, gamma_bar, p)
        self.assertEqual(len(params), 5)

if __name__ == '__main__':
    unittest.main()
