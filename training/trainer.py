"""
Core Training Logic
Handles model training loop, validation, and checkpointing.
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm
from data.dataset_factory import get_combined_dataset
from data.transforms import get_training_transforms
from models.segmentation import HybridRoadSafetyModel, RoadSafetySegmentationModel
from models.loss import CombinedLoss 

def train_model(args):
    """
    Main training function.
    """
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Data
    print("Loading 5 Hugging Face Datasets + Local Data...")
    config = {
        'target_size': (640, 640),
        'hf_datasets': [
            # 1. Road Segmentation (General Road) - Map Class 1 to Road(1)
            {'name': "UniqueData/roads-segmentation-dataset", 'split': "train[:200]", 'label_map': {1: 1}},
            
            # 2. Amine Pothole (Specific Potholes) - Map Class 1 to Pothole(2)
            {'name': "amineeeee123/pothole-segmentation", 'split': "train[:200]", 'label_map': {1: 2}},
            
            # 3. Road Issues (Cracks/Potholes) - Map Class 1? to Pothole(2) 
            # (Note: Heuristic mapping, verify if strict mapping needed)
            {'name': "Programmer-RD-AI/road-issues-detection-dataset", 'split': "train[:200]", 'label_map': {1: 2, 2: 2}},
            
            # 4. Keremberke Pothole (YOLO style masks)
            {'name': "keremberke/pothole-segmentation", 'split': "train[:200]", 'label_map': {1: 2}},
            
            # 5. Road Detection (Bnsapa) - Map Drivable Area(1) to Road(1)
            {'name': "bnsapa/road-detection", 'split': "train[:200]", 'label_map': {1: 1}}
        ],
        
        # Local Data (Priority)
        'local_pothole_dir': "datasets/pothole_extracted" if os.path.exists("datasets/pothole_extracted") else None,
        'negative_sample_dir': "datasets/Urban Civic Issues Image Dataset Potholes and Garb/pothole/no" 
                               if os.path.exists("datasets/Urban Civic Issues Image Dataset Potholes and Garb/pothole/no") else None
    }
    
    # Safety wrapper to continue even if some datasets fail download
    try:
        dataset = get_combined_dataset(config, transform=get_training_transforms())
    except Exception as e:
        print(f"Critical Error loading datasets: {e}")
        return

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Total training samples: {len(dataset)}")

    # 3. Model
    # Training the custom backbone
    model = RoadSafetySegmentationModel(num_classes=3, backbone=args.backbone) 
    model = model.to(device)
    
    # 4. Optimizer & Loss
    # High weight for Pothole (2) to handle imbalance
    class_weights = torch.tensor([0.5, 1.0, 10.0]).to(device) 
    criterion = CombinedLoss(alpha=class_weights, num_classes=3)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 5. Loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for images, masks, _ in loop:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=rf"{loss.item():.4f}")
            
        avg_loss = running_loss / len(loader) if len(loader) > 0 else 0
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Checkpoint
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_multi_dataset_epoch_{epoch+1}.pth"))
        
    print("Training Complete. Model saved.")
