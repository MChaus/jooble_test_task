from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class GlobalStatistic(ABC):
    '''
    GlobalStatistic is supposed to evaluate some statistic that need entire 
    dataset to be calculated. It may evaluate it iteratively if dataset is too
    big and devided in chunks. In this case the estimation looks like learning,
    when every new piece of data makes estimation better.
    '''
    @abstractmethod
    def __init__(self, feature_codes):
        # Set of features codes we are interested in
        self._feature_codes = feature_codes
        # Value of statistic for each feature
        self._statistic = dict.fromkeys(self._feature_codes, None)
        # Number of records we observed for each feature
        self._n = dict.fromkeys(self._feature_codes, 0)
    
    def calculate_statistic_from_file(self, file_path, chunk_size):
        file_reader = pd.read_csv(file_path, 
                                  iterator=True, 
                                  chunksize=chunk_size, 
                                  delimiter='\t')
        while True:
            try:
                chunk = next(file_reader)
                self.improve_estimation(chunk)
            except StopIteration:
                break
        return self.value

    def _prepare_chunk(self, chunk):
        '''
        Return two columns - feature code with numeric values
        '''        
        features = chunk['features'].str.split(',', n=1, expand=True)
        features = features[features[0].isin(self._feature_codes)]
        return features
    
    @abstractmethod
    def improve_estimation(self, chunk):
        '''
        Recalculate estimations with impact of the chunk.
        
        Every new piece of information makes our estimations better.
        '''
        pass
    
    @property
    @abstractmethod
    def value(self):
        pass
    
    
class GlobalMean(GlobalStatistic):
    
    def __init__(self, feature_codes):
        super().__init__(feature_codes)
    
    def improve_estimation(self, chunk):
        chunk = self._prepare_chunk(chunk)
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
        '''Set initial mean value for each feature. 
        '''
        self._statistic[feature] = mini_chunk.mean(axis = 0)
        self._n[feature] += mini_chunk.shape[0]
    
    @property
    def value(self):
        return self._statistic
    

class GlobalStd(GlobalStatistic):
    
    def __init__(self, mean, feature_codes):
        super().__init__(feature_codes)
        self._mean = mean
    
    def improve_estimation(self, chunk):
        chunk = self._prepare_chunk(chunk)
        for feature in self._feature_codes:
            mini_chunk = chunk[chunk[0] == feature]
            mini_chunk = mini_chunk[1].str.split(',', expand=True)
            mini_chunk = mini_chunk.to_numpy(dtype='int64')
            
            k = mini_chunk.shape[0]
            if k == 0:
                continue    # no values for current feature in the chunk
            
            if self._statistic[feature] is None:
                self._init_statistic(feature, mini_chunk)
                continue
            
            # Calculate iterative uncorrected sample variance 
            square_sum = np.square(mini_chunk - self._mean[feature]).sum(axis = 0) 
            self._statistic[feature] += square_sum / self._n[feature]
            self._statistic[feature] *= self._n[feature] / (self._n[feature] + k)
            self._n[feature] += k
            
    def _init_statistic(self, feature, mini_chunk):
        '''Set initial uncorrected sample variance for each feature
        '''
        self._n[feature] += mini_chunk.shape[0]
        self._statistic[feature] = np.square(mini_chunk - self._mean[feature]).mean(axis = 0) 
        
    @property
    def variance(self):
        ''' Return corrected sample variance
        '''
        variance = dict()
        for feature in self._feature_codes:
            if self._statistic[feature] is None:
                variance[feature] = None
                continue
            variance[feature] = self._statistic[feature] * self._n[feature] / (self._n[feature] - 1)
        return variance
    
    @property
    def value(self):
        ''' Return corrected std
        '''
        std = dict()
        for feature in self._feature_codes:
            if self._statistic[feature] is None:
                std[feature] = None
                continue
            std[feature] = self._statistic[feature] * self._n[feature] / (self._n[feature] - 1)
            std[feature] = np.sqrt(std[feature])
        return std