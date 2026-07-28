"""
Multimodal Decision-Level Fusion
Combines data from Vision, Speed, and Weather modules to produce a final risk score.
"""

def fuse_multimodal_risk(vision_risk, speed_kmh, weather_condition):
    """
    Decision-Level Fusion Strategy:
    Final Risk = w_v * VisionRisk + w_s * SpeedRisk + w_w * WeatherRisk
    
    Args:
        vision_risk (float): Risk from segmentation module (0.0 - 1.0)
        speed_kmh (float): Vehicle speed in km/h
        weather_condition (int/str): Weather condition index or name
                                     0: Clear, 1: Rain/Light, 2: Severe
    
    Returns:
        float: Final combined risk score (0.0 - 1.0)
        dict: Breakdown of individual risk components
    """
    
    # 1. Normalize Speed Risk
    # Assume 120 km/h is max risk reference
    speed_risk = min(speed_kmh / 120.0, 1.0)
    
    # 2. Normalize Weather Risk
    # Map simple weather indices or strings to risk 0-1
    if isinstance(weather_condition, str):
        weather_map = {'clear': 0.0, 'rain': 0.5, 'snow': 0.8, 'fog': 1.0}
        weather_risk = weather_map.get(weather_condition.lower(), 0.0)
    else:
        # Assuming int: 0=Safe, 1=Cautious, 2=Danger
        weather_risk = min(weather_condition / 2.0, 1.0)
        
    # 3. Fusion Weights (Domain Knowledge)
    # Vision is primary (50%), Speed (30%), Weather (20%)
    w_vision = 0.5
    w_speed = 0.3
    w_weather = 0.2
    
    final_risk = (w_vision * vision_risk) + (w_speed * speed_risk) + (w_weather * weather_risk)
    
    # Ensure final clip
    final_risk = min(final_risk, 1.0)
    
    return final_risk, {
        "vision_component": vision_risk,
        "speed_component": speed_risk,
        "weather_component": weather_risk
    }

def get_risk_level_label(risk_score):
    """
    Returns text label and color for a given risk score.
    """
    if risk_score < 0.3:
        return "Safe", (0, 255, 0) # Green
    elif risk_score < 0.7:
        return "Caution", (0, 255, 255) # Yellow
    else:
        return "Danger", (0, 0, 255) # Red
