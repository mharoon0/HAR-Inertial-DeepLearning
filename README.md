# Wearable Sensor Human Activity Recognition (HAR): Feature-Based MLP vs. End-to-End 1D-CNN

This repository contains a comprehensive comparative machine learning project developed in **Python** using **PyTorch**. The study investigates the performance trade-offs, architecture dynamics, and computational efficiencies between two distinct deep learning methodologies applied to the benchmark **UCI Human Activity Recognition (HAR) Dataset**.

---

## 📋 Project Overview
The objective of this project is to classify six distinct human physical activities (**Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, and Laying**) using continuous 3-axial linear acceleration and angular velocity signals captured at 50 Hz from a waist-mounted Samsung Galaxy S II smartphone.

The project evaluates two architectural paradigms:
1. **Model A (Baseline MLP):** A Multi-Layer Perceptron trained on a vector of 561 pre-calculated, expertly engineered time and frequency domain features.
2. **Model B (Challenger 1D-CNN):** A 1D-Convolutional Neural Network that accepts raw, tri-axial inertial sensor windows directly, utilizing localized temporal filters to extract abstract feature hierarchies from scratch.

---

## 📊 Key Performance Insights

| Model Architecture | Test Accuracy | Macro F1-Score | Primary Strengths |
| :--- | :---: | :---: | :--- |
| **MLP (Feature-Based)** | **94.7%** | **0.95** | Superior at distinguishing static postures (Sitting vs. Standing) via gravity-vector orientation features. |
| **1D-CNN (End-to-End)** | **93.1%** | **0.93** | Outperforms in capturing high-frequency dynamic gait patterns (e.g., *Walking Upstairs*). |

### Structural Error Trade-off (Confusion Matrix Breakdown):
* **Static Postures:** The MLP misclassified **77 total samples** between Sitting and Standing. The 1D-CNN struggled significantly more on this boundary with **176 total errors**, demonstrating that explicitly isolated gravity features (e.g., $angle(X, gravityMean)$) provide stronger baseline orientation anchors than raw localized temporal convolutions.
* **Dynamic Postures:** The 1D-CNN successfully outperformed the MLP on dynamic gait profiles, accurately identifying **464 Walking Upstairs samples** compared to the MLP’s **431**, validating its proficiency in automated temporal feature extraction without mathematical smoothing.

---

## 🛠️ Tech Stack & Key Libraries
* **Framework:** PyTorch (Neural Network construction, customized DataLoader pipelines, and tensor management)
* **Exploratory Data Analysis:** Scikit-Learn (Principal Component Analysis & Data Scaling)
* **Data Processing & Analytics:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn

---

## 🏗️ Model Architectures

### 1. Multi-Layer Perceptron (MLP)
* **Input Layer:** 561 neurons (Normalized via `StandardScaler`)
* **Hidden Layers:** Fully Connected Layer (128 neurons, ReLU activation) $\rightarrow$ Fully Connected Layer (64 neurons, ReLU activation)
* **Regularization/Output:** Softmax-ready Linear Layer with 6 classification outputs.

### 2. 1D-Convolutional Neural Network (1D-CNN)
* **Data Dimension Transformation:** Input arrays shaped at `(N, 128, 9)` are permuted to a channel-first alignment of `(N, 9, 128)` to meet PyTorch framework conventions.
* **Feature Extraction Blocks:** 3 stacked 1D-Convolutional blocks scaling channels from $64 \rightarrow 128 \rightarrow 256$.
* **Hyperparameters:** Initial kernel size of 5 for capturing atomic movements (~0.1s tracking windows), down-scaling to a kernel size of 3 in deeper layers. 
* **Regularization & Classification Head:** Integrated MaxPool1D layers, Dropout ($p = 0.5$) to prevent neuron co-adaptation, and a dense integration layer (128 neurons) prior to final classification.

---

## 📈 Visualizations

### Feature Space Separation via Principal Component Analysis (PCA)
<img width="1188" height="790" alt="PCA_Analysis" src="https://github.com/user-attachments/assets/871747d4-af50-4037-b785-f4ad5caffb9b" />

* **Observation:** PCA projection onto a 2D space reveals distinct, linearly separable clusters for dynamic/gait-related activities, while revealing highly complex structural overlaps between static positions (Sitting vs. Standing).

### Model Error Profile Breakdown (Confusion Matrices)
The confusion matrices highlight exactly where the model classifications excel and where structural misinterpretations occur on the independent 30% test set splits.

#### 1. Multi-Layer Perceptron (MLP) — Final Test Accuracy: 94.7%
<img width="1500" height="1200" alt="MLP_Confusion_Matrix_Test" src="https://github.com/user-attachments/assets/2604a208-2b0c-442a-89a2-57bdef355629" />

  * ***Static Strengths:*** Effectively isolated the boundaries for static postures, containing only ***77 total misclassifications*** between Sitting and Standing. This demonstrates that explicitly isolated gravity vector inputs provide strong orientation anchors.
  * ***Dynamic Bottlenecks:*** Underperformed slightly on highly rhythmic gait modifications, accurately mapping ***431 samples*** for *Walking Upstairs*.


#### 2. 1D-Convolutional Neural Network (1D-CNN) — Final Test Accuracy: 93.1%
<img width="1500" height="1200" alt="CNN_Confusion_Matrix_Test" src="https://github.com/user-attachments/assets/67b524bf-a659-493f-bf5d-256cc3dd800f" />

  * ***Static Vulnerabilities:*** Struggled heavily to identify absolute spatial positions from scratch, yielding a total of **176 errors** between Sitting and Standing. This highlights that localized temporal filters cannot naturally substitute orientation invariants without a much longer tracking window.
  * ***Dynamic Mastery:*** Outperformed the MLP on complex continuous movements, correctly predicting **464 samples**
