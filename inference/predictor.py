"""
Core Inference Engine
Handles loading models and running prediction on images or video frames.
"""

import torch
import numpy as np
import cv2
from PIL import Image
from models.segmentation import HybridRoadSafetyModel, PretrainedRoadSegmentationModel, RoadSafetySegmentationModel
from data.transforms import preprocess_sample

class RoadSafetyPredictor:
    """
    Main inference class.
    """
    def __init__(self, model_path=None, use_pretrained=False, force_cpu=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
        print(f"Predictor using device: {self.device}")
        
        self.target_size = (640, 640) # Standardize
        
        # Load Model Strategy
        if model_path and use_pretrained:
            print("Loading Hybrid Model (Pretrained Road + Trained Pothole)...")
            self.model = HybridRoadSafetyModel(trained_model_path=model_path)
        elif use_pretrained:
            print("Loading Pretrained DeepLabV3+ (Road Only)...")
            self.model = PretrainedRoadSegmentationModel()
        elif model_path:
            print("Loading Custom Trained Model...")
            self.model = RoadSafetySegmentationModel(num_classes=3)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
             raise ValueError("Must provide model_path or enable use_pretrained")
             
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        """
        Predict on a single image (path or PIL or numpy).
        
        Returns:
            mask (np.ndarray): (H, W) class labels
            probs (np.ndarray): (3, H, W) raw probabilities (optional)
        """
        # Handle input types
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        original_size = image.size
        
        # Preprocess
        # We assume mask is None since this is inference
        img_tensor, _, _ = preprocess_sample(image, 0, self.target_size) 
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(img_tensor)
            
        # Post-process
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Resize back to original
        # Use Nearest Neighbor to keep class IDs integer
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
        
        return pred_mask
