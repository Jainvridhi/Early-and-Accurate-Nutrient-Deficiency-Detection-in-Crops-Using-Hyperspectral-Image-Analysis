🌿 Early and Accurate Nutrient Deficiency Detection in Crops

Using Hyperspectral Image Analysis and Ensemble Deep Learning
This project provides an AI-driven, hardware-free solution to detect nutrient deficiencies in crops at an early stage.
By integrating Hyperspectral Imaging (HSI) for pre-symptomatic detection and RGB imaging for disease diagnosis, the system serves as a comprehensive diagnostic-to-prescription pipeline for both farmers and researchers
## 📌 Overview

This project presents a software-driven framework for **early detection of 
crop nutrient deficiencies** — Nitrogen (N), Phosphorus (P), and Potassium (K) 
— using **Hyperspectral Imaging (HSI)** and an **Ensemble of Deep Learning 
models**, without requiring expensive hardware or lab equipment.

The system detects deficiencies **10–15 days before visible symptoms appear**, 
enabling timely intervention and potentially reducing annual crop losses by 
**20–40%**.

> 📄 Research Paper: *Early and Accurate Nutrient Deficiency Detection in Crops 
> Using Hyperspectral Image Analysis and Ensemble Deep Learning*  
> Vridhi Jain, Mitanshi, Mansha — Bharati Vidyapeeth's College of Engineering, 
> New Delhi

---

## 🎯 Key Results

| Model | Accuracy | F1 (Healthy) | F1 (Partial) | F1 (Deficient) |
|---|---|---|---|---|
| ResNet50 | 65.09% | 0.622 | 0.000 | 0.731 |
| DenseNet121 | 72.33% | 0.742 | 0.000 | 0.779 |
| **CustomCNN** | **93.40%** | **0.958** | **0.781** | **0.946** |
| Ensemble + TTA | 92.00% | 0.946 | 0.654 | 0.939 |

- ⚡ **148 ms inference time** on CPU — no GPU required
- 🌱 Detects deficiencies **10–15 days** before visual symptoms
- 🗣️ **Bilingual interface** — English and Hindi
- 📊 **LIME + SHAP** explainability built in

---

## 🏗️ Project Structure
EANDD/
│
├── data/
│   ├── plantvillage/          # RGB images (54,000+)
│   ├── hyperleaf2024/         # Hyperspectral samples (100,000+)
│   └── preprocessed/          # PCA-compressed outputs
│
├── models/
│   ├── custom_cnn.py          # Custom CNN architecture
│   ├── resnet50_adapter.py    # ResNet50 with 1x1 adapter block
│   ├── densenet121_refiner.py # DenseNet121 with feature refiner
│   └── ensemble.py            # Weighted Soft Voting + TTA
│
├── preprocessing/
│   ├── pca_compression.py     # 15-component PCA pipeline
│   ├── normalization.py       # Z-score normalization
│   ├── augmentation.py        # Rotation, flip, blur, brightness
│   └── fusion.py              # HSI + RGB multimodal fusion
│
├── explainability/
│   ├── lime_analysis.py       # LIME heatmaps per class
│   └── shap_bands.py          # Global PCA band importance
│
├── app/
│   ├── streamlit_app.py       # Main Streamlit dashboard
│   ├── farmer_view.py         # Simple bilingual farmer interface
│   ├── expert_view.py         # Detailed researcher interface
│   └── report_generator.py    # Automated PDF reports (ReportLab)
│
├── results/
│   ├── confusion_matrix.png
│   ├── bar_accuracy.png
│   ├── per_class_metrics.png
│   └── lime_explanations/
│
├── requirements.txt
├── README.md


---

## 🧠 Model Architecture

### 1. Custom CNN — Spectral Specialist ⭐ Best Model
- Input: PCA-compressed 15-channel hyperspectral patches (128×128×15)
- 3× Conv2D layers (32 → 64 → 128 filters) + BatchNorm + ReLU
- Residual shortcut connections for smooth gradient flow
- MaxPooling2D + Dropout (p=0.5)
- Dense + Softmax output (Healthy / Partial / Deficient)
- Optimizer: Adam (η = 0.0001), Gradient Clipping (Clipnorm = 1.0)

### 2. ResNet50 — Morphology Specialist
- 1×1 Adapter Block converts 15-channel input → 3 channels
- Pretrained ImageNet weights, fine-tuned on HyperLeaf2024
- Best for structural and morphological leaf patterns

### 3. DenseNet121 — Feature Refiner
- Feature Refiner Block (3×3 + 1×1 convolutions)
- Dense connections preserve fine spectral details
- Best for texture-level spectral variation

### 4. Ensemble + TTA
- Weighted Soft Voting across all three models
- Test-Time Augmentation for prediction stability
- Partial class precision improved to 0.850

---

## 🔄 Pipeline
Raw Hyperspectral Cube (150+ bands)
↓
Otsu Masking + Z-score Normalization
↓
PCA Compression (150+ bands → 15 components, 95.2% variance retained)
↓
┌─────────────┬──────────────┬─────────────┐
│  CustomCNN  │   ResNet50   │ DenseNet121 │
└─────────────┴──────────────┴─────────────┘
↓
Weighted Soft Voting + TTA
↓
Final Prediction: Healthy / Partial / Deficient
↓
LIME + SHAP Explanation → Streamlit Dashboard → PDF Report

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/EANDD.git
cd EANDD

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📋 Requirements

```txt
tensorflow>=2.10.0
torch>=1.13.0
torchvision>=0.14.0
scikit-learn>=1.1.0
numpy>=1.23.0
pandas>=1.5.0
opencv-python>=4.6.0
streamlit>=1.15.0
lime>=0.2.0
shap>=0.41.0
dask>=2022.10.0
reportlab>=3.6.0
matplotlib>=3.6.0
seaborn>=0.12.0
Pillow>=9.3.0
joblib>=1.2.0
```

---

## 🚀 Usage

### Run the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

### Train Models
```bash
# Train CustomCNN
python models/custom_cnn.py --epochs 50 --batch_size 32

# Train Ensemble
python models/ensemble.py --mode train
```

### Run Inference
```bash
python models/ensemble.py --mode predict --input your_image.npy
```

### Generate LIME Explanations
```bash
python explainability/lime_analysis.py --image your_image.npy
```

---

## 📊 Datasets

| Dataset | Type | Samples | Details |
|---|---|---|---|
| HyperLeaf2024 | Hyperspectral | 100,000+ | 400–1000 nm, 150+ bands, drone + handheld |
| PlantVillage | RGB | 54,000+ | 14 crop types, 26 disease conditions, 256×256 |

Data split: **70% train / 15% validation / 15% test** with stratified sampling.

---

## 🖥️ Streamlit Dashboard

The app features a **dual-interface design**:

**🌾 Farmer View**
- Upload leaf image → instant prediction
- Fertilizer recommendations (e.g. 50 kg/hectare urea)
- Auto-generated PDF report
- Full English + Hindi support

**🔬 Expert View**
- Detailed spectral analysis
- LIME heatmaps per class
- SHAP PCA band importance charts
- Per-class confidence breakdown
- Confusion matrix and model comparison

---

## 🔍 Explainability

This project uses **LIME** and **SHAP** to ensure biological validity:

- **680 nm** — Chlorophyll absorption, iron deficiency marker (SHAP = 0.22)
- **550–700 nm** — Red-edge region, nitrogen depletion (SHAP = 0.18)
- **1400 nm** — Water absorption band, phosphorus stress indicator

LIME heatmaps confirm the model attends to biologically meaningful regions
rather than spurious image artifacts.

---

## 🌍 Impact

- 🇮🇳 Supports **India's Digital Agriculture Mission**
- 👨‍🌾 Accessible to **140M+ farmers** without technical knowledge
- 📉 Potential to reduce crop losses by **20–40%** (FAO estimates)
- ⏱️ Detects deficiencies **10–15 days early** before visible symptoms
- 💻 Runs on **commodity CPUs** — no GPU needed

---

## 👩‍💻 Authors

**Vridhi Jain** — [GitHub](https://github.com/Jainvridhi)  
**Mitanshi**  
**Mansha**  

Department of Information Technology  
Bharati Vidyapeeth's College of Engineering, New Delhi  
Affiliated to Guru Gobind Singh Indraprastha University, Delhi

---



---

## 🙏 Acknowledgements

- Dr. Alka Leeka — Project Supervisor
- HyperLeaf2024 Dataset Contributors
- PlantVillage Dataset — Penn State University
- Food and Agriculture Organization (FAO) — crop loss statistics

---

## 📬 Citation

If you use this work, please cite:

```bibtex
@article{jain2025eandd,
  title={Early and Accurate Nutrient Deficiency Detection in Crops 
         Using Hyperspectral Image Analysis and Ensemble Deep Learning},
  author={Jain, Vridhi and Mitanshi and Mansha},
  journal={Bharati Vidyapeeth's College of Engineering},
  year={2025}
}
```

---

