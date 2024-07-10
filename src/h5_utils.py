import h5py
import numpy as np
from typing import List, Tuple, Union 

class WriterWithIntegrity:
    def __init__(self, integrity_cols: List[str],
                        target_col: str):
        self.integrity_cols = integrity_cols
        self.target_col = target_col

    def get_train_feats_only(self, df):
        feature_columns = [col for col in df.columns if col not in \
                           ['date_id', self.target_col, *self.integrity_cols]]
        return feature_columns
    
    def save_metadata(self, df, filepath)->None:
        date_ids = sorted(df['date_id'].unique())
        column_names = self.get_train_feats_only(df)
        
        with h5py.File(filepath, 'w') as f:
            
            f.create_dataset('date_ids', data=date_ids)
            f.create_dataset('column_names', data=column_names)

    def write_daily_hdf5(self, df, filename)->None:
        with h5py.File(filename, 'w') as f:
            
            integrity_group = f.create_group('integrity_cols')
            for column in self.integrity_cols:
                integrity_group.create_dataset(column, 
                                               data=df[column].to_numpy())
    
            f.create_dataset('data/target', 
                             data=df[self.target_col].to_numpy())
    
            features_group = f.create_group('data/features')
            
            feature_columns = self.get_train_feats_only(df)
            for column in feature_columns:
                features_group.create_dataset(column, 
                                              data=df[column].to_numpy())

def load_metadata(filepath)->np.ndarray:
    with h5py.File(filepath, 'r') as f:
        return f['date_ids'][:]

def load_daily_minimal(date_id: Union[int, str], 
                       folder_daily_h5,
                       features_selected: Tuple[str]=(),
                       float32: bool = False)->(np.ndarray,np.ndarray):
    '''Train optimized usage. Ommit feature names and integrity data'''
    filepath=f'{folder_daily_h5}/{date_id}.h5'
    with h5py.File(filepath, 'r') as f:
        # Load the target column
        target = f['data']['target'][:]

        features_group = f['data']['features']
        if features_selected:
            values_lists = [features_group[name][:] for name in features_selected]
        else:
            values_lists = [features_group[name][:] for name in features_group.keys()]
        
        if float32:
            features = np.array(values_lists,  dtype=np.float32).T 
        else:
            features = np.array(values_lists).T
            
    return features, target

def stacked_daily_data(date_ids: Union[List[int], List[str]], 
                       folder_daily_h5,
                       features_selected: Tuple[str]=(),
                       float32: bool = False)->(np.ndarray,np.ndarray):
        features_stacked = []
        labels_stacked = []
        for date_id in date_ids:
            daily_features, daily_labels = load_daily_minimal(date_id, 
                                                              folder_daily_h5,
                                                              features_selected,
                                                              float32)
            
            features_stacked.append(daily_features)
            labels_stacked.append(daily_labels)
        
        features_stacked = np.vstack(features_stacked)
        labels_stacked = np.hstack(labels_stacked)
        return features_stacked, labels_stacked