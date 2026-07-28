"""
Visualization Utilities
Handles creation of segmentation overlays, charts, and video frames.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class RoadSafetyVisualizer:
    """
    Helper for drawing overlays and risk dashboards.
    """
    def __init__(self):
        # Color Map (BGR for OpenCV, RGB for PIL)
        # 0: Background (Black/Transparent)
        # 1: Road (Green)
        # 2: Pothole (Red)
        self.colors = {
            0: [0, 0, 0],
            1: [0, 255, 0],
            2: [255, 0, 0] 
        }

    def create_overlay(self, image, mask, alpha=0.4):
        """
        Overlays segmentation mask on the original image.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is valid
        if image is None: return None
            
        # Create colored mask
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for cls, color in self.colors.items():
            colored_mask[mask == cls] = color
            
        # Resize mask to image if needed (safety check)
        if (h, w) != (image.shape[0], image.shape[1]):
            colored_mask = cv2.resize(colored_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Blend
        overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
        return overlay

    def draw_dashboard(self, frame, risk_score, speed, weather_condition, risk_breakdown=None):
        """
        Draws the risk dashboard on the frame (HUD style).
        """
        h, w, _ = frame.shape
        
        # Dashboard Config
        bg_color = (0, 0, 0)
        text_color = (255, 255, 255)
        
        # Draw header bar
        cv2.rectangle(frame, (0, 0), (w, 80), bg_color, -1)
        
        # 1. Overall Risk
        from fusion.decision_fusion import get_risk_level_label
        label, color = get_risk_level_label(risk_score)
        # Convert RGB color to BGR for OpenCV
        color_bgr = (color[2], color[1], color[0])
        
        cv2.putText(frame, f"RISK SCORE: {risk_score:.2f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color_bgr, 2)
        cv2.putText(frame, f"LEVEL: {label}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
        
        # 2. Metadata (Speed, Weather)
        cv2.putText(frame, f"SPEED: {speed:.1f} km/h", (350, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)
        weather_str = str(weather_condition) if isinstance(weather_condition, str) else ["Clear", "Rain", "Severe"][min(weather_condition, 2)]
        cv2.putText(frame, f"WEATHER: {weather_str}", (350, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)
        
        # 3. Component Breakdown (Optional)
        if risk_breakdown:
            x_start = 650
            cv2.putText(frame, "FACTORS:", (x_start, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Vis: {risk_breakdown['vision_component']:.2f}", (x_start, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            cv2.putText(frame, f"Spd: {risk_breakdown['speed_component']:.2f}", (x_start + 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            cv2.putText(frame, f"Wth: {risk_breakdown['weather_component']:.2f}", (x_start + 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        return frame
