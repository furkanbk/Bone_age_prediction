# Advancing Handbone Maturity Estimation with CNNs and Decision Fusion

## Overview
This repository contains the implementation for the paper **"Advancing Handbone Maturity Estimation: Comparative Analysis and Novel Approaches Using CNNs and Decision Fusion"**. The project focuses on improving skeletal maturity estimation from hand radiographs using convolutional neural networks (CNNs) and decision fusion mechanisms. We compare state-of-the-art models and propose novel enhancements to boost accuracy.

## Key Contributions
- **Comparative Analysis of CNN Architectures**: Evaluating Inception-v4, CNN with CBAM attention, ResNet50 with transfer learning, and a baseline CNN.
- **Decision Fusion Mechanism**: Incorporating carpal bone regions to improve prediction accuracy.
- **Dynamic Data Augmentation**: Addressing dataset imbalance with a novel augmentation method.
- **Gender Information in Model Training**: Assessing its impact on model performance.

## Project Structure
```
├── DatasetPreprocessing.ipynb  # Data preprocessing notebook
├── Models_and_Training.ipynb   # Model training and evaluation notebook
├── ProjectPaper.pdf  # Detailed explanation of the project
└── README.md                    # Project documentation
```

## Dataset
We use a dataset containing 14,036 hand radiographs collected from two institutions for bone age assessment. The dataset includes:
- Train: 12,611 images
- Validation: 1,425 images
- Test: 200 images

### Data Preprocessing
The **DatasetPreprocessing.ipynb** notebook handles:
- Image enhancement using **CLAHE (Contrast Limited Adaptive Histogram Equalization)**.
- Label removal via binary masking.
- Image resizing to 256x256 pixels.
- Data augmentation to address class imbalance.

## Models Implemented
Seven different architectures were implemented in **Models_and_Training.ipynb**:
1. **Baseline CNN** - Simple CNN with 4 layers and gender input.
2. **Inception-v4** - An advanced Inception architecture for image regression.
3. **Attention-CNN (CBAM)** - CNN with channel and spatial attention mechanisms.
4. **Decision Fusion** - Combining predictions from full images and cropped carpal bone regions.
5. **ResNet50 (Transfer Learning)** - Utilizing a pre-trained ResNet50 model for comparison.
6. **ResNet50 + Gender** - Evaluating the impact of gender information.
7. **ResNet50 + Gender + Trainable Layers** - Allowing first 10 layers to be trainable.

## Model Evaluation
- **Loss Function**: Root Mean Squared Error (RMSE).
- **Performance Metrics**: Mean Absolute Error (MAE), training time, inference speed, and model size.

### Best Performing Model: Decision Fusion
- **Test MAE**: **9.24 months**
- **Inference Time**: **11.4s per image**
- **Advantages**: Improved accuracy by incorporating carpal bone regions.

## Installation & Usage
### Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy
- OpenCV
- Matplotlib
- Pandas

### Setup
Clone the repository:
```bash
git clone https://github.com/your-username/handbone-maturity-estimation.git
cd handbone-maturity-estimation
```
Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Notebooks
1. **Dataset Preprocessing**:
   - Open `DatasetPreprocessing.ipynb` and run all cells.
   - This will generate three preprocessed datasets (`raw`, `clahe`, `clahe_masked`).

2. **Model Training and Evaluation**:
   - Open `Models_and_Training.ipynb` and run all cells.
   - This will train all models and generate performance statistics.

## Results & Insights
- **Decision fusion** improved accuracy beyond individual CNN models.
- **Attention mechanisms (CBAM)** provided a performance boost without adding significant computational overhead.
- **Gender information** had varying impact across models, sometimes degrading accuracy.
- **Balanced dataset** training resulted in smoother convergence and improved generalization.

## Future Work
- Extending training epochs to further improve accuracy.
- Testing additional architectures such as Vision Transformers (ViTs).
- Exploring real-world deployment strategies for clinical use.

## Authors
- **Berat Furkan Kocak** - University of Padova
- **Onur Bacaksiz** - University of Padova

For inquiries, please contact: `beratfurkan.kocak@studenti.unipd.it`

