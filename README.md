<div align="center">

# 🚦 Multimodal Road Safety Risk Assessment System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)](#)

*An advanced, AI-driven system for evaluating road safety risks in real-time by fusing computer vision with multimodal environmental data.*

</div>

---

## 🌟 Overview

The **Multimodal Road Safety Risk Assessment System** is a deep learning-based pipeline designed to identify and quantify road hazards. By combining semantic segmentation (vision) with external metadata such as vehicle speed and weather conditions, the system calculates a holistic "risk score" in real-time, providing actionable insights for drivers or autonomous driving systems.

## ✨ Key Features

- **🧠 Real-Time Semantic Segmentation**: Uses state-of-the-art DeepLabV3+ (with ResNet-18 or MobileNet-V2 backbones) to identify road boundaries, obstacles, and lanes.
- **🔗 Multimodal Data Fusion**: Intelligently fuses vision-based risks with external factors like speed and weather conditions.
- **📊 Dynamic Dashboard Visualization**: Generates real-time, augmented overlays on video feeds with a risk-scoring dashboard.
- **⚡ CPU & GPU Support**: Optimized for both high-performance GPU clusters and edge-device CPU deployment.
- **🖼️ Flexible Inference**: Run inference seamlessly on single images, bulk image directories, or full video streams.

---

## 🏗️ System Architecture

1. **Vision Module (`models/`)**: Performs pixel-perfect semantic segmentation to detect hazards.
2. **Fusion Module (`fusion/`)**: Calculates a holistic risk score using the segmented mask, speed, and weather data.
3. **Inference & Visualization (`inference/`)**: Overlays the segmentation mask and a dynamic data dashboard onto the original frame.

---

## 🚀 Getting Started

### 1. Prerequisites

Make sure you have Python 3.8+ installed.

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/SathyaKrishna-M/Multimodal-Road-Safety-Risk-Assessment-System.git
cd Multimodal-Road-Safety-Risk-Assessment-System

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### 🎯 Inference

The inference script supports single images, directories, and video files. 

**Run on a single image:**
```bash
python run_inference.py --image_path path/to/image.jpg --hybrid
```

**Run on a directory of images:**
```bash
python run_inference.py --input_dir path/to/images_folder --hybrid
```

**Run on a video stream:**
```bash
python run_inference.py --video_path path/to/video.mp4 --hybrid
```

*Note: The `--hybrid` flag uses the recommended pre-trained model architecture. Add `--force_cpu` if you don't have a CUDA-enabled GPU.*

### 🛠️ Training

To train the segmentation model on your own dataset:

```bash
python train_model.py --epochs 20 --batch_size 8 --backbone resnet18
```

---

## 📂 Project Structure

```text
├── models/                  # Deep learning models (Segmentation, Loss)
├── fusion/                  # Multimodal decision fusion & risk scoring
├── inference/               # Predictor and Dashboard Visualizer logic
├── training/                # Training pipelines and utilities
├── train_model.py           # Main entry point for training
├── run_inference.py         # Main entry point for inference
├── requirements.txt         # Project dependencies
└── .gitignore               # Ignored files (datasets, logs, caches)
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<div align="center">
<i>Built with ❤️ for safer roads.</i>
</div>
