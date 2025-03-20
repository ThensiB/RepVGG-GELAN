# RepVGG-GELAN: Enhanced GELAN with VGG-STYLE ConvNets for Brain Tumour Detection

## Overview
RepVGG-GELAN is a novel deep learning model designed for accurate and efficient brain tumour detection in medical images. It integrates RepVGG, a reparameterized convolutional approach, into the YOLO framework, enhancing speed and precision. The model also leverages Generalized Efficient Layer Aggregation Network (GELAN) to further improve feature extraction and detection performance.

## Features
- Enhanced YOLO architecture by incorporating RepVGG for optimized object detection.
- Efficient feature extraction by utilizing GELAN for improved accuracy and speed.
- State-of-the-Art Performance by Achieving superior results compared to existing RCS-YOLO models.
- Optimized for medical imaging, tailored for brain tumour detection using deep learning.

## Dataset
The model is trained and evaluated on the **Brain Tumour Detection 2020 (Br35H) dataset** from Kaggle. This dataset consists of 701 medical images labeled for tumour detection.

## Installation
To set up the environment and run the model, follow these steps:
```bash
# Clone the repository
git clone https://github.com/ThensiB/RepVGG-GELAN.git
cd RepVGG-GELAN

# Install dependencies
pip install -r requirements.txt
```

## Training the Model
The model is implemented using **PyTorch** and trained on **Google Colab** with an **NVIDIA RTX 3090 GPU**.
```bash
# Train the model
python train.py --data data.yaml --epochs 150 --batch-size 8 --img-size 640
```

## Model Architecture
RepVGG-GELAN combines:
- **RepVGG Blocks**: Reparametrized convolutional networks for efficient learning.
- **GELAN Blocks**: Spatial Pyramid Pooling and multi-scale feature extraction.
- **ADown Modules**: Asymmetric downsampling for better localization.

## Performance
| Model | Precision | Recall | AP50 | AP50:95 | Parameters (M) |
|--------|------------|------------|------------|------------|-----------------|
| RCS-YOLO | 0.936 | 0.945 | 0.946 | 0.729 | 45.7 |
| YOLOv8 | 0.973 | 0.909 | 0.957 | 0.733 | 30.1 |
| **RepVGG-GELAN** | **0.982** | **0.890** | **0.970** | **0.723** | **25.4** |

## Evaluation Metrics
The model's performance is measured using:
- **Precision & Recall**: Measures the accuracy of predictions.
- **mAP (mean Average Precision)**: Evaluates model accuracy over different IoU thresholds.
- **FLOPs (Floating Point Operations)**: Assesses computational efficiency.

## Inference
Run the following command to perform inference on test images:
```bash
python detect.py --weights best.pt --source test_images/
```

## Citation
If you use this work, please cite:
```
@article{balakrishnan2024repvgg-gelan,
  title={RepVGG-GELAN: Enhanced GELAN with VGG-STYLE ConvNets for Brain Tumour Detection},
  author={Balakrishnan, Thennarasi and Sengar, Sandeep Singh},
  journal={arXiv preprint arXiv:2405.03541},
  year={2024}
}
```

## Contact
For queries, reach out to:
- **Thennarasi Balakrishnan** - thennarasibalakrishnan@gmail.com

## Acknowledgments
This research was conducted at **Cardiff Metropolitan University**. Special thanks to contributor and the open-source community for their valuable resources.

