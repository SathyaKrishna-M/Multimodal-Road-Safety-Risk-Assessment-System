"""
Vision-Based Risk Scorer
Calculates risk scores purely from visual segmentation data (Vision Module).
"""

import numpy as np

def calculate_vision_risk(segmentation_mask, class_weights=None):
    """
    Computes a risk score based *only* on segmentation analysis (pixel density).
    
    Args:
        segmentation_mask (np.ndarray): 2D array of class labels (H, W).
        class_weights (dict): Dictionary mapping class IDs to risk weights.
                              Default: {2: 1.0} (Pothole = High Risk)
        
    Returns:
        float: Normalized vision risk score (0.0 to 1.0).
    """
    if class_weights is None:
        # Default weights
        class_weights = {
            1: 0.1,  # Road (Low risk baseline)
            2: 1.0,  # Pothole (High risk)
        }

    h, w = segmentation_mask.shape
    total_pixels = h * w
    
    vision_risk = 0.0
    classes_present = np.unique(segmentation_mask)
    
    for cls in classes_present:
        if cls in class_weights:
            # Count pixels for this class
            count = np.sum(segmentation_mask == cls)
            # Normalized density
            density = count / total_pixels
            
            # Linear contribution with saturation
            # e.g., if 20% of screen is potholes, it's very high risk
            contribution = density * class_weights[cls] * 5.0 
            vision_risk += contribution
            
    # Clip to 0-1 range
    return min(vision_risk, 1.0)
