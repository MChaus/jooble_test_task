from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class GlobalStatistic(ABC):
    
    @abstractmethod
    def __init__(self, feature_codes, file_path, chunk_size):
        # Set of features codes we are interested in
        self._feature_codes = feature_codes
        self._file_path = file_path
        self._chunk_size = chunk_size
        self._file_reader = pd.read_csv(self._file_path, 
                                        iterator=True, 
                                        chunksize=self._chunk_size, 
                                        delimiter='\t')
        # Value of statistic for each feature
        self._statistic = None
        self._step_count = 0
    
    def calculate_statistic(self):
        while True:
            try:
                self._step_count += 1
                chunk = next(self._file_reader)
                chunk = self._prepare_chunk(chunk)
                self._improve_estimation(chunk)
            except StopIteration:
                break
        return self._statistic

    @abstractmethod
    def _prepare_chunk(self, chunk):
        pass
    
    @abstractmethod
    def _improve_estimation(self, chunk):
        pass
    
    
class GlobalMean(GlobalStatistic):
    
    def __init__(self, feature_codes, file_path, chunk_size):
        super().__init__(feature_codes, file_path, chunk_size)
        # Number of records for each feature
        self._n = dict.fromkeys(self._feature_codes, 0)
        self._statistic =  dict.fromkeys(self._feature_codes, None)
        
    def _prepare_chunk(self, chunk):
        '''Return two columns - feature code with numeric values'''        
        
        features = chunk['features'].str.split(',', n=1, expand=True)
        features = features[features[0].isin(self._feature_codes)]
        return features
    
    def _improve_estimation(self, chunk):
        for feature in self._feature_codes:
            mini_chunk = chunk[chunk[0] == feature]
            mini_chunk = mini_chunk[1].str.split(',', expand=True)
            mini_chunk = mini_chunk.to_numpy(dtype='int64')
            
            k = mini_chunk.shape[0]
            if k == 0:
                continue    # no values for current feature in the chunk
            
            if self._statistic[feature] is None:
                self._init_statistic(feature, mini_chunk)
                continue    # calculate the average first time
            
            # Calculate iterative mean
            mini_sum = mini_chunk.sum(axis = 0)
            self._statistic[feature] += mini_sum / self._n[feature]
            self._statistic[feature] *= self._n[feature] / (self._n[feature] + k)
            self._n[feature] += k
            
    def _init_statistic(self, feature, mini_chunk):
        # Set initial mean value for each feature
        self._statistic[feature] = mini_chunk.mean(axis = 0)
        self._n[feature] += mini_chunk.shape[0]