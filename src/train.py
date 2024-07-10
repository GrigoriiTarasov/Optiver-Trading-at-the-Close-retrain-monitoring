import numpy as np
def drop_target_nan(features: np.ndarray,
                    target: np.ndarray)-> (np.ndarray, np.ndarray):
    
    mask = ~np.isnan(target)
    
    return features[mask], target[mask]