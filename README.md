🌿 Early and Accurate Nutrient Deficiency Detection in Crops

Using Hyperspectral Image Analysis and Ensemble Deep Learning
This project provides an AI-driven, hardware-free solution to detect nutrient deficiencies in crops at an early stage.
By integrating Hyperspectral Imaging (HSI) for pre-symptomatic detection and RGB imaging for disease diagnosis, the system serves as a comprehensive diagnostic-to-prescription pipeline for both farmers and researchers

🚀 Key Features👨‍🌾 

Farmer View (Actionable Insights)Instant Diagnosis:
Upload standard RGB leaf photos to detect diseases using a robust ensemble of ResNet50 and Custom CNNs.Simulated Nutrient Analysis:
Provides early warnings of nutrient stress before they become visible to the eye.
Smart Prescription Engine: Delivers detailed chemical and organic treatment protocols, including active ingredients like Potassium Nitrate ($KNO_3$) or Urea.
Multilingual Support: Automated reports and recommendations are available in English and Hindi.Dosage Calculator: A dynamic tool to calculate the exact fertilizer weight required based on field size and deficiency severity.

🔬 Researcher View (Deep Analysis)Real-time .TIFF Processing: 
Direct support for raw hyperspectral .tiff files (e.g., $6.6$MB files with $150+$ bands).
Dimensionality Reduction: Utilizes a real-time Principal Component Analysis (PCA) pipeline to compress spectral data into a $3$-component feature space for rapid inference.
Explainable AI (XAI): Integrated LIME heatmaps to visualize the spectral "areas of interest" driving the model's decision.
Spectral Visualization: Reconstructs PCA components into false-color images to highlight physiological stress.

📊 Performance Metrics
The final ensemble framework achieves state-of-the-art results by combining spatial features from RGB data and spectral signatures from HSI.
Metric,Result
Final Test Accuracy,94.03%
Final Test Loss,0.1781
Severe Deficiency (0.0) Precision,0.99
Mild Deficiency (1.0) F1-Score,0.97

🛠️ Tech StackFrontend: 
Streamlit.Deep Learning: TensorFlow, Keras, ResNet50, DenseNet121, Custom CNN.
Data Processing: Scikit-learn (PCA, StandardScaler), OpenCV, Joblib.
Geospatial/Spectral Tools: Tifffile, Rasterio.
XAI: LIME (Local Interpretable Model-agnostic Explanations

📂 Dataset InformationThe models were trained on a heterogeneous multimodal dataset:
RGB: PlantVillage Dataset (Disease classification).
Hyperspectral: HyperLeaf2024, NASA AVIRIS, USDA, USGS Crop Library, and WHU-Hyperspectral.

🏗️ Project Structure

/Mini Project
├── /data
│   └── /processed          # Precomputed PCA pipelines & encoders
├── /models
│   ├── nutrient_hyper_v2.h5 # Champion HSI model
│   └── plant_disease_rgb.h5 # Champion RGB model
├── /src
│   ├── app.py              # Unified Streamlit Interface
│   ├── inference.py        # Model loading and XAI logic
│   └── report_generator.py  # PDF and Audio report generation [cite: 121]
└── requirements.txt
