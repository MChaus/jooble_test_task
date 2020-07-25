import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import unittest
import numpy as np
from global_statistics import GlobalMean

mean_test = {
    '2': np.array([104, 204, 304, 404], dtype='float64'),
    '3': np.array([1, 2, 5, 50, 500, 5000], dtype='float64'),
    '4': None
}

class GlobalMeanTest(unittest.TestCase):
    def test_global_mean(self):
        file_path = os.path.join(current_dir, 'test_data', 'test.tsv')
        sizes = [1, 5, 100]
        features_of_interest = {'2', '3', '4'}

        for chunk_size in sizes:
            with self.subTest(chunk_size=chunk_size):
                mean = GlobalMean(
                    features_of_interest,
                    file_path,
                    chunk_size
                )
                mean_train = mean.calculate_statistic()
                self.assert_equal_dicts(mean_train, mean_test)
                
    def assert_equal_dicts(self, dict1, dict2):
        self.assertEqual(dict1.keys(), dict2.keys())
        for feature in dict1:
            if dict1[feature] is None:
                self.assertEqual(dict1[feature], dict2[feature])
                continue
            np.testing.assert_almost_equal(dict1[feature], dict2[feature])
        
if __name__ == '__main__':
    unittest.main(verbosity=3)