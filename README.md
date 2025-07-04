# Medical Insurance Charges Prediction (RBF Regressor)

This project applies a **Radial Basis Function (RBF) Network** combined with **Kernel Ridge Regression (KRR)** to predict medical insurance costs using demographic and health-related features like age, BMI, smoking status, and region. The model captures complex non-linear patterns through RBF transformations and clustering, showing the power of kernel-based methods in real-world healthcare analytics. Built as part of a **Problem-Based Learning** module at NMAM Institute of Technology.

---

## 🚀 Overview

* **Tech Stack:**

  * `Python` — Core language for ML
  * `pandas` — Data handling
  * `scikit-learn` — KRR, RBF kernel, KMeans, preprocessing
  * `NumPy` — Numerical operations
  * `matplotlib / seaborn` — EDA, correlation plots, scatter & line charts
    
---

## 🧩 Project Structure & Flow

### 1️⃣ Data Preprocessing

* **Features:**
  
  * `age` — Age of the individual
  * `sex` — Gender (encoded: male=1, female=0)
  * `bmi` — Body Mass Index
  * `children` — Number of dependents
  * `smoker` — Smoking status (1 = smoker, 0 = non-smoker)
  * `region` — Geographic region (label encoded)
  * `charges` — Insurance cost (target)

* **Steps:**
  
  * Handle missing values — median for numerics, mode for categoricals.
  * Encode categorical columns with Label Encoding.
  * Normalize numerics with **StandardScaler** for consistent scaling.
  * Split data: 80% train, 20% test.

---

### 2️⃣ RBF Regressor (Kernel Ridge Regression)

**Core Idea:**
- Uses **RBF kernel** to map input to higher dimensions.
- Employs **Kernel Ridge Regression (KRR)** for stable learning.
- Integrates **KMeans Clustering** (45 clusters, Elbow Method) for better local pattern extraction.

**Key Parameters:**
- `alpha = 0.001` — Regularization to avoid overfitting.
- `gamma = 0.08` — Controls the spread of the RBF.
- `kernel = 'rbf'` — Kernelized regression.

---

### 3️⃣ Evaluation

* **Metrics:**
  
  * Mean Squared Error (MSE)
  * Total Squared Error (TSE)
  * R² Score

---

## 📊 Results


| Metric | Training | Testing |
|--------|----------|---------|
| MSE | 0.1301 | 0.1506 |
| R² Score | 86.79% | 85.78% |

---

## 🎯 Purpose

This project demonstrates the use of **RBF networks with Kernel Ridge Regression** to tackle a real-world regression problem in insurance cost estimation. It highlights how feature engineering (clustering + kernel methods) can improve model performance in complex datasets.

---

## 📂 References

- Kaggle Medical Insurance Dataset.
- Claesen et al. — Fast prediction with SVM RBF kernels.
- Izonin et al. — SGTM Neural Structures with RBF for insurance cost prediction.
- Veerappermal et al. — RBFNs in healthcare image analysis.

---

## 📌 To Do / Extensions

- Add GridSearch for hyperparameter tuning (`alpha`, `gamma`).
- Compare with other kernel methods (SVM with RBF, GPR).
- Build a Streamlit/Flask app for real-time insurance charge prediction.
- Expand to multi-output or anomaly detection in claims.

---

## 🏫 Institution

NMAM Institute of Technology, Nitte (Deemed to be University)

---

## 📜 License

This work is for **educational use only** and not intended for production or commercial deployment.

---

**Stay curious, and may your RBFs cover all clusters while your errors stay minimal!**
