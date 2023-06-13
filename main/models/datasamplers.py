"""
Module for data set classes to be used in the models 
and generic preparation functions for the sampling in the folktables
experiments
"""
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from torch.utils.data import DataLoader, Dataset

class TabularDataSet(Dataset):
    def __init__(self, X, y_1, y_2):
        self.X = X.copy()
        self.y1 = y_1
        self.y2 = y_2
        
    def __len__(self):
        return len(self.y1)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y1[idx], self.y2[idx]
    
class TabularDataSetSim(Dataset):
    def __init__(self, X, y_1, y_2, missing_split):
        self.X = X.copy()
        self.y1 = y_1
        self.y2 = y_2
        self.missing_split = missing_split
        
    def __len__(self):
        return len(self.y1)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y1[idx], self.y2[idx], self.missing_split[idx]
    
def prepare_sim_data(full_features, 
                     full_labels, 
                     split_seed,
                     missing_data=np.array([0,0.25,0.5,0.75,0.9]), 
                     missing_data_seed=42):
    
    # set up scaler
    numeric_cols = ['SCHL', 'WKHP', 'JWMNP']
    other_cols = full_features.drop(columns = numeric_cols).columns

    onehot_ = OneHotEncoder()
    scaler_ = StandardScaler()

    transformer = ColumnTransformer(
        transformers=[('categoricals', onehot_, other_cols), 
                    ('numerical', scaler_, numeric_cols)
                    ], 
                    remainder='passthrough', 
                    sparse_threshold=0)

    # Split 
    X_train, X_test, y_train, y_test = train_test_split(full_features,
                                                        full_labels, 
                                                        test_size=0.2,
                                                        random_state=split_seed)
    
    # Scale
    transformer.fit(X_train)
    X_scaled = np.float32(transformer.transform(X_train))
    X_test_scaled = np.float32(transformer.transform(X_test))

    print(X_scaled.shape)
    
    # Sample missing data vectors
    np.random.seed(missing_data_seed)
    missing_matrix = np.random.uniform(0,1,size=(X_train.shape[0], 
                                                 len(missing_data)))
    layover_matrix = np.repeat(missing_data.reshape(-1,len(missing_data)),
                               X_train.shape[0],
                               axis=0)
    weight_mask = np.where(missing_matrix > layover_matrix, 1.0, 0.0)

    
    # combine into a new datasets
    dataset_dict = {}
    
    for col_idx, missing_data_val in enumerate(missing_data):
        dataset_dict[missing_data_val] = {}

        weights = weight_mask[:, col_idx]

        train_load = DataLoader(TabularDataSetSim(X_scaled, 
                                                  np.float32(y_train.iloc[:,0]).reshape(-1,1),
                                                  np.float32(y_train.iloc[:,1]).reshape(-1,1), 
                                                  np.float32(weights.reshape(-1,1))),
                                batch_size=128,
                                shuffle=True)

        train_all_pred = (X_scaled,
                          np.float32(y_train.iloc[:,0]).reshape(-1,1),
                          np.float32(y_train.iloc[:,1]).reshape(-1,1))
        
        test_load = (X_test_scaled,
                     np.float32(y_test.iloc[:,0]).reshape(-1,1),
                     np.float32(y_test.iloc[:,1]).reshape(-1,1))
        
        dataset_dict[missing_data_val]['train'] = train_load
        dataset_dict[missing_data_val]['test'] = test_load
        dataset_dict[missing_data_val]['train_all'] = train_all_pred

    return dataset_dict
