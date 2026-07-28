"""
Main Training Entry Point
Usage:
    python train_model.py --epochs 20 --force_cpu
    python train_model.py --backbone mobilenet_v2 --batch_size 8
"""
import argparse
from training.trainer import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Road Safety Model")
    
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU training')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'mobilenet_v2'], help='Model backbone')
    
    args = parser.parse_args()
    
    train_model(args)
