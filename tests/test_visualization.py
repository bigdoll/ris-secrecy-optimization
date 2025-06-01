import unittest
import numpy as np
import matplotlib.pyplot as plt
from visualization import Plotter, Plot3DPositions

class TestPlotter(unittest.TestCase):
    
    def setUp(self):
        self.x_val = np.arange(0, 10, 1)
        self.data_series = [
            {
                'data': np.random.rand(10),
                'label': 'Series 1',
                'color': 'blue',
                'marker': 'o',
                'line_width': 3
            },
            {
                'data': np.random.rand(10),
                'label': 'Series 2',
                'color': 'red',
                'marker': 'x',
                'line_width': 3
            }
        ]
        self.plotter = Plotter(self.x_val, self.data_series, x_type='power', plot_type='Data Rate')

    def test_plot_results(self):
        # This test is more about ensuring no exceptions are raised
        self.plotter.plot_results()

    def test_set_labels(self):
        self.plotter._set_labels()
        self.assertEqual(plt.gca().get_xlabel(), "Maximum available Transmit Power (dBm)")
        self.assertEqual(plt.gca().get_ylabel(), "Data Rate (bps/Hz)")

    def test_apply_smoothing(self):
        data = np.random.rand(10)
        smoothed_data = self.plotter._apply_smoothing(data)
        self.assertEqual(len(smoothed_data), 10)

class TestPlot3DPositions(unittest.TestCase):
    
    def setUp(self):
        self.RIS = np.array([0, 0, 10])
        self.Tx = [np.array([i, i, i]) for i in range(1, 5)]
        self.Rx_B = np.array([10, 0, 0])
        self.Rx_E = np.array([0, 10, 0])
        self.plotter_3d = Plot3DPositions(self.RIS, self.Tx, self.Rx_B, self.Rx_E)

    def test_plot(self):
        # This test is more about ensuring no exceptions are raised
        self.plotter_3d.plot()

    def test_plot_point(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        point = np.array([1, 1, 1])
        self.plotter_3d._plot_point(ax, point, 'Test Point', 'black', 'o')
        self.assertEqual(len(ax.collections), 1)  # Check that one point is plotted

if __name__ == '__main__':
    unittest.main()
