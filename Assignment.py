"""
- There is a limitation in this code that there must be at least 8 instances for the Gaussian distribution check-up to be done.
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.stats as stats
import numpy as np
from typing import List, Dict, Callable

def normalize_data(input_data: List[Dict[str, any]]) -> List[Dict[str, any]]:
    """
    Normalize numerical data in the input list of dictionaries.
    
    Args:
        input_data (List[Dict[str, any]]): List of dictionaries representing features.

    Returns:
        List[Dict[str, any]]: List of dictionaries with normalized numerical features.
    """
    new_list = []
    for feature in input_data:
        if feature['type'] == 'numerical':
            normalization_method = get_normalization_method(feature['value'])
            new_feature = {'value': normalization_method(feature['value']), 'type': feature['type']}
            new_list.append(new_feature)
    return new_list

def get_normalization_method(value: List[float]) -> Callable[[List[float]], List[float]]:
    """
    Determine the appropriate normalization method based on the data distribution.
    
    Args:
        value (List[float]): List of numerical values.

    Returns:
        Callable[[List[float]], List[float]]: Normalization function.
    """
    if is_gaussian(value):
        return standard_scale
    elif is_bounded(value):
        return min_max_scale
    elif is_skewed(value):
        return log_transform
    else:
        return lambda x: x  # No normalization

def standard_scale(data: List[float]) -> List[float]:
    """
    Standardize data using StandardScaler.
    
    Args:
        data (List[float]): List of numerical values.

    Returns:
        List[float]: List of standardized values.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1)).flatten()
    return scaled_data.tolist()

def min_max_scale(data: List[float]) -> List[float]:
    """
    Scale data to a fixed range using MinMaxScaler.
    
    Args:
        data (List[float]): List of numerical values.

    Returns:
        List[float]: List of scaled values.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1)).flatten()
    return scaled_data.tolist()

def log_transform(data: List[float]) -> List[float]:
    """
    Apply log transformation to data.
    
    Args:
        data (List[float]): List of numerical values.

    Returns:
        List[float]: List of log-transformed values.
    """
    log_transformed_data = np.log1p(np.array(data))
    return log_transformed_data.tolist()

def is_gaussian(data: List[float]) -> bool:
    """
    Check if data follows a Gaussian distribution.
    
    Args:
        data (List[float]): List of numerical values.

    Returns:
        bool: True if data is Gaussian, False otherwise.
    """
    k2, p_value = stats.normaltest(data)
    return p_value > 0.05

def is_bounded(data: List[float]) -> bool:
    """
    Check if data is bounded between 0 and 1.
    
    Args:
        data (List[float]): List of numerical values.

    Returns:
        bool: True if data is bounded, False otherwise.
    """
    min_val, max_val = np.min(data), np.max(data)
    return 0 <= min_val and max_val <= 1

def is_skewed(data: List[float]) -> bool:
    """
    Check if data is skewed.
    
    Args:
        data (List[float]): List of numerical values.

    Returns:
        bool: True if data is skewed, False otherwise.
    """
    skewness = stats.skew(data)
    return abs(skewness) > 1

if __name__ == "__main__":
    input_data = [{"value": [123456,32334,23442,46464,64444,663450,38555,565658], "type": "numerical"},
                {"value": [500, 50, 5, 25, 100, 1000, 5000, 1500], "type": "numerical"},
                {"value": [10, 20, 30, 40, 50, 70, 90, 100], "type": "numerical"},
                {"value": [0.01, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 0.1], "type": "numerical"},
                {"value": "Some Text", "type": "text"}]  # Non-numerical data ignored  

    normalized_data = normalize_data(input_data) 
    print(normalized_data) 