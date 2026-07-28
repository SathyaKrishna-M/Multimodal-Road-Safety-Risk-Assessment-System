"""
Main Inference Entry Point
Unified script for running inference on:
1. Single Images
2. Directories
3. Videos
4. Hugging Face Datasets (Evaluation)

Usage:
    python run_inference.py --input_dir my_images --force_cpu --hybrid
    python run_inference.py --video_path road.mp4
"""

import argparse
import os
import glob
from tqdm import tqdm
import cv2
import numpy as np

from inference.predictor import RoadSafetyPredictor
from inference.visualizer import RoadSafetyVisualizer
from fusion.risk_scorer import calculate_vision_risk
from fusion.decision_fusion import fuse_multimodal_risk

def process_image(predictor, visualizer, img_path, output_dir, args):
    """Run pipeline on a single image"""
    filename = os.path.basename(img_path)
    
    # 1. Prediction (Vision)
    mask = predictor.predict(img_path)
    
    # 2. Risk Calculation (Fusion)
    # Simulate metadata for static images
    speed = 60.0 # Placeholder
    weather = "Clear" # Placeholder
    
    vision_risk = calculate_vision_risk(mask)
    final_risk, breakdown = fuse_multimodal_risk(vision_risk, speed, 0) # 0=Clear
    
    # 3. Visualization
    # Read original for overlay
    original = cv2.imread(img_path)
    if original is None: return
    
    overlay = visualizer.create_overlay(original, mask)
    dashboard = visualizer.draw_dashboard(overlay.copy(), final_risk, speed, weather, breakdown)
    
    # Save
    cv2.imwrite(os.path.join(output_dir, f"result_{filename}"), dashboard)

def process_video(predictor, visualizer, video_path, output_dir, args):
    """Run pipeline on video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out_path = os.path.join(output_dir, "result_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {video_path}...")
    
    # Simulating dynamic metadata
    current_speed = 50.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Predict
        # Convert BGR (OpenCV) to RGB (PIL/Model)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = predictor.predict(frame_rgb)
        
        # 2. Fusion
        # Simple simulation: speed varies slightly
        current_speed += np.random.uniform(-1, 1)
        current_speed = np.clip(current_speed, 0, 120)
        
        vision_risk = calculate_vision_risk(mask)
        final_risk, breakdown = fuse_multimodal_risk(vision_risk, current_speed, 0)
        
        # 3. Visualize
        overlay = visualizer.create_overlay(frame, mask)
        dashboard = visualizer.draw_dashboard(overlay, final_risk, current_speed, "Clear", breakdown)
        
        out.write(dashboard)
        
    cap.release()
    out.release()
    print(f"Video saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Road Safety Inference")
    
    # Input options
    parser.add_argument('--input_dir', type=str, help='Directory of images')
    parser.add_argument('--image_path', type=str, help='Single image path')
    parser.add_argument('--video_path', type=str, help='Video file path')
    
    # Model config
    parser.add_argument('--model_path', type=str, default='results/model_epoch_1.pth', help='Path to trained model')
    parser.add_argument('--use_pretrained', action='store_true', help='Use pretrained DeepLabV3+')
    parser.add_argument('--hybrid', action='store_true', help='Use hybrid model (Recommended)')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/inference', help='Output directory')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Core Modules
    # Hybrid flag overrides standard flags
    use_pretrained = args.use_pretrained or args.hybrid
    model_path = args.model_path if args.hybrid else (args.model_path if not args.use_pretrained else None)
    
    predictor = RoadSafetyPredictor(
        model_path=model_path, 
        use_pretrained=use_pretrained,
        force_cpu=args.force_cpu
    )
    
    visualizer = RoadSafetyVisualizer()
    
    # Routing
    if args.video_path:
        process_video(predictor, visualizer, args.video_path, args.output_dir, args)
    
    elif args.input_dir:
        images = glob.glob(os.path.join(args.input_dir, "*"))
        images = [f for f in images if f.lower().endswith(('.jpg', '.png'))]
        print(f"Found {len(images)} images.")
        for img in tqdm(images):
            process_image(predictor, visualizer, img, args.output_dir, args)
            
    elif args.image_path:
        process_image(predictor, visualizer, args.image_path, args.output_dir, args)
        
    else:
        print("Please specify --input_dir, --image_path, or --video_path")

if __name__ == "__main__":
    main()
