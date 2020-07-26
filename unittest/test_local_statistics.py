import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import unittest
import numpy as np
from local_statistics import MaxIndex, MaxAbsMeanDiff, ZScore
import pandas as pd

mean_test = {
    '2': np.array([104, 204, 304, 404], dtype='float64'),
    '3': np.array([1, 2, 5, 50, 500, 5000], dtype='float64'),
    '4': None
}

std_test = {
    '2': np.sqrt([7.5, 7.5, 7.5, 7.5], dtype='float64'),
    '3': np.sqrt([0, 0, 11, 1100, 110000, 11000000], dtype='float64'),
    '4': None
}

mamd_test = pd.Series([4, 3, 2, 1, 0, 1, 2, 3, 4,
                       0, 4000, 3000, 2000, 1000, 0, 1000, 2000, 3000, 4000, 5000], 
                      dtype='float64',
                      name='max_abs_mean_diff')

mean_z_test = {
    '2': np.array([0, 0, 2, 8], dtype='float64'),
    '3': np.array([0, 0, 8, 16], dtype='float64'),
    '4': None
}

std_z_test = {
    '2': np.array([4, 4, 4, 4], dtype='float64'),
    '3': np.array([8, 16, 8, 16], dtype='float64'),
    '4': None
}

z_score_test = np.array([[1,    2,  1,  2],
                         [1,    0,  1,  0],
                         [1,    0,  1,  0],
                         [1,    0,  1,  0],
                         [0,    0,  0,  0],
                         [-1,   0,  -1,  0],
                         [-1,   0,  -1,  0],
                         [-1,   0,  -1,  0],
                         [-1,   -2, -1,  -2],
                         [1,    1,  1,  1],
                         [1,    1,  1,  1],
                         [0,    0,  0,  0],
                         [-1,   -1, -1,-1],
                         [-1,   -1, -1,-1]],
                         dtype='float64')

class LocalStatisticTest(unittest.TestCase):
    def test_local_max_index(self):
        file_path = os.path.join(current_dir, 'test_data', 'test.tsv')
        features_of_interest = {'2', '3', '4'}
        max_index = MaxAbsMeanDiff(features_of_interest, mean_test)
        
        df = pd.read_csv(file_path, delimiter='\t')
        df = max_index.calculate_statistic(df)
        
        pd.testing.assert_series_equal(df['max_abs_mean_diff'], mamd_test)
        
    def test_local_z_score(self):
        file_path = os.path.join(current_dir, 'test_data', 'z_test.tsv')
        features_of_interest = {'2', '3', '4'}
        z_score = ZScore(features_of_interest, mean_z_test, std_z_test)
        
        df = pd.read_csv(file_path, delimiter='\t')
        df = z_score.calculate_statistic(df)
        
        score = self._vectorize(df['stand'])
        np.testing.assert_almost_equal(score, z_score_test)
        
    def _vectorize(self, column):
        features = column.str.split(',', expand=True)
        features = np.array(features, dtype='float64')
        return features
                
        
if __name__ == '__main__':
    unittest.main(verbosity=3)