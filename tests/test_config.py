import unittest
from config import SystemConfig

class TestSystemConfig(unittest.TestCase):
    
    def setUp(self):
        self.config = SystemConfig()

    def test_initial_ris_state(self):
        self.assertEqual(self.config.ris_state, 'active')

    def test_initial_opt_state(self):
        self.assertEqual(self.config.opt_state, 'ee')

    def test_noise_power_spectral_density(self):
        self.assertAlmostEqual(self.config.N0_dBm_Hz, -174.0, places=1)

    def test_db_to_linear(self):
        self.assertEqual(self.config.db_to_linear(10), 10.0)

    def test_dbm_to_watts(self):
        self.assertAlmostEqual(self.config.dbm_to_watts(0), 0.001, places=6)

    def test_calculate_noise_power_variance(self):
        self.assertAlmostEqual(self.config.calculate_noise_power_variance(self.config.N0, self.config.BW0), 5.47e-20, places=22)

    def test_calculate_channel_estimation_error_variance(self):
        variance = self.config.calculate_channel_estimation_error_variance(self.config.N0, self.config.BW, self.config.SNR_E_linear)
        self.assertAlmostEqual(variance, 5.47e-13, places=15)

    def test_set_ris_state(self):
        self.config.set_ris_state('passive')
        self.assertEqual(self.config.ris_state, 'passive')
        self.assertEqual(self.config.a, 1)
        with self.assertRaises(ValueError):
            self.config.set_ris_state('invalid_state')

    def test_set_opt_state(self):
        self.config.set_opt_state('sr')
        self.assertEqual(self.config.opt_state, 'sr')
        with self.assertRaises(ValueError):
            self.config.set_opt_state('invalid_state')

    def test_compute_static_power_consumption(self):
        power_consumption = self.config.compute_static_power_consumption(100)
        self.assertAlmostEqual(power_consumption, 0.503, places=3)

if __name__ == '__main__':
    unittest.main()
