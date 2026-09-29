import numpy as np

def extract_image_features(image):
    """Extract spatial domain features from image matrix."""
    mean_val = float(np.mean(image))
    std_val = float(np.std(image))
    max_val = float(np.max(image))
    min_val = float(np.min(image))
    
    return np.array([mean_val, std_val, max_val, min_val])
