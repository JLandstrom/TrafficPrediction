import unittest

from pandas.util.testing import assert_frame_equal

import TrafficPreprocessor
import pandas as pd

class TestTrafficPreprocessor(unittest.TestCase):
    """Set up any testvariables in setup"""
    def setUp(self):
        self.param = pd.DataFrame({'DetectorID':[1064],'VehicleClassID':[0],'Timestamp':['2018-01-01'],'Flow':[60],'NoVehicles':[1]})
        self.negativeObjectParam = -2
        self.characterObjectParam = 'k'

    def tearDown(self):
        pass

    def test_DefaultConstructor(self):
        preprocessor = TrafficPreprocessor.TrafficPreprocessor()
        assert_frame_equal(preprocessor.PreProcess(self.param, 1), self.param)

    def test_InvalidMdHandler(self):
        preprocessor = TrafficPreprocessor.TrafficPreprocessor(mdHandler='KN12N')
        self.assertRaises(ValueError, lambda: preprocessor.PreProcess(self.param, 0))


if __name__ == '__main__':
    unittest.main()