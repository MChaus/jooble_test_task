import os,sys
from local_statistics import MaxAbsMeanDiff, ZScore
from global_statistics import GlobalMean, GlobalStd
import pandas as pd
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(current_dir, 'data', 'train.tsv')
test_path = os.path.join(current_dir, 'data', 'test.tsv')

def get_mean_std(df, features_of_interest):
    mean = GlobalMean(features_of_interest)
    mean.improve_estimation(df)
    
    std = GlobalStd(mean.value, features_of_interest)
    std.improve_estimation(df)
    
    return mean.value, std.value

def add_local_stats(df, mean, std, features_of_interest):
    max_index = MaxAbsMeanDiff(features_of_interest, mean)
    z_score = ZScore(features_of_interest, mean, std)
    
    local_statistics = [max_index, z_score]
    
    for statistic in local_statistics:
        df = statistic.add_statistic(df)
    
    return df

def prepare_df(df):
    stand = df['stand'].str.split(',', expand=True)
    stand = stand.to_numpy(dtype='float64')
    
    for col in range(stand.shape[1]):
        col_name = 'feature_2_stand_' + str(col)
        df[col_name] = stand[:, col]
    
    df = df.rename(columns={
            'max_index':'max_feature_2_index',
            'max_abs_mean_diff':'max_feature_2_abs_mean_diff'
            })
    df = df.drop(['stand'], axis = 1)
    df = df.drop(['features'], axis=1)
    return df
    
def main():
    
    features_of_interest = {'2',}    
    test_df = pd.read_csv(test_path, delimiter='\t')
    train_df = pd.read_csv(train_path, delimiter='\t')

    mean, std = get_mean_std(train_df, features_of_interest)
    test_df = add_local_stats(test_df, mean, std, features_of_interest)
    test_df = prepare_df(test_df)
    test_df.to_csv('test_proc.tsv', sep='\t', index=False)

    print("File has been genereted.")
    
if __name__ == '__main__':
    main()