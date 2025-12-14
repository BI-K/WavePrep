from preprocessing import AbstractImputer
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import joblib

imputers_that_need_fitting = ['iterative_imputer', 'hinrichs_paper']

class HinrichsPaperImputer(AbstractImputer):

    def impute(self, processed_data, channel_name, path) -> np.ndarray:

        channel = processed_data[channel_name].copy()
        if np.any(np.isnan(channel)):
            isnan = np.isnan(channel)
            n = len(channel)

            # Forward-fill up to 3 consecutive missing values
            i = 0
            while i < n:
                if isnan[i]:
                    # Start of missing block
                    start = i
                    while i < n and isnan[i]:
                        i += 1
                    end = i
                    gap = end - start
                    channel[start:start + 3] = channel[start - 1]
                else:
                    i += 1


            # If any np.nan remain, use IterativeImputer
            if np.any(np.isnan(channel)):

                channel_names = list(processed_data.keys())
                channel_idx = next((i for i, name in enumerate(channel_names) if name == channel_name), None)
                imputer = joblib.load(path)

                processed_data[channel_name] = channel
                processed_data_array = np.array(list(processed_data.values())).T
                imputed_data = imputer.transform(processed_data_array)
                channel = imputed_data[:, channel_idx].flatten()

        return channel


    

def is_imputer_that_needs_split(imputer_name: str) -> bool:
    return imputer_name in imputers_that_need_fitting

def create_imputer(method: str = 'hinrichs_paper') -> AbstractImputer:
    """
    Factory function to create an imputer instance based on the specified method.
    
    Args:
        method: Imputation method to use (e.g., 'hinrichs')
        
    Returns:
        Instance of AbstractImputer subclass
    """
    if method == 'hinrichs_paper':
        return HinrichsPaperImputer()
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    

def train_and_save_imputer(data: np.ndarray, imputer_name: str, save_path: str) -> None:
    """
    Train the specified imputer on the provided channel data and save it to disk.
    
    Args:
        channel_data: 2D array where each row is a time series for a subject
        imputer_name: Name of the imputer to train
        save_path: Path to save the trained imputer
    """

    data = np.array(data)
    data = data.reshape(-1, data.shape[2])  # Combine all samples for training

    if imputer_name == 'iterative_imputer' or imputer_name == 'hinrichs_paper':

        imputer = IterativeImputer(estimator=None, max_iter=10, random_state=0, sample_posterior=False)
        imputer.fit(data)
        # Save the trained imputer using joblib or pickle
        joblib.dump(imputer, save_path)
    else:
        raise ValueError(f"Unknown imputer name: {imputer_name}")