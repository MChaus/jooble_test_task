from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class LocalStatistic(ABC):
    
    @abstractmethod
    def __init__(self, feature_codes):
        self._feature_codes = feature_codes
        
    @abstractmethod
    def calculate_statistic(self, chunk):
        pass
    
    
class MaxIndex(LocalStatistic):
    def __init__ (self, feature_codes):
        super().__init__(feature_codes)
    
    def calculate_statistic(self, chunk):
        chunk['max_index'] = chunk.apply(self._max_index, axis=1)
        return chunk
            
    def _max_index(self, row):
        features = row['features'].split(',')
        if features[0] in self._feature_codes:
            features = np.array(features[1:], dtype='int64')
            return np.argmax(features)
        else:
            return np.nan
        
class MaxAbsMeanDiff(LocalStatistic):
    def __init__ (self, feature_codes, mean):
        super().__init__(feature_codes)
        self._max_index = MaxIndex(feature_codes)
        self._mean = mean
    
    def calculate_statistic(self, chunk):
        chunk = self._max_index.calculate_statistic(chunk)
        chunk['max_abs_mean_diff'] = chunk.apply(self._max_abs_mean_diff, axis=1)
        return chunk
        
    def _max_abs_mean_diff(self, row):
        features = row['features'].split(',')
        feature_code = features[0]
        if feature_code in self._feature_codes:
            max_id = row['max_index']
            mean = self._mean[feature_code][max_id]
            
            value = abs(mean - int(features[max_id + 1]))
            return value
        else:
            return np.nan

class ZScore(LocalStatistic):
    def __init__ (self, feature_codes, mean, std):
        super().__init__(feature_codes)
        self._max_index = MaxIndex(feature_codes)
        self._mean = mean
        self._std = std
        
    def calculate_statistic(self, chunk):
        chunk["stand"] = chunk.apply(self._z_score, axis=1)
        return chunk
    
    def _z_score(self, row):
        features = row['features'].split(',')
        feature_code = features[0]
        if feature_code in self._feature_codes:
            mean = self._mean[feature_code]
            std = self._std[feature_code]
            
            features = np.array(features[1:], dtype='float64')
            z_score = (features - mean) / std
            return ','.join(str(value) for value in z_score)
        else:
            return np.nan

            
        
        
        