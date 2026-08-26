# 🏭 Predictive Maintenance System

## 📌 Overview
This is a Machine Learning Graduation Project aimed at predicting equipment failure and diagnosing the specific type of failure using the **AI4I 2020 Predictive Maintenance Dataset**. 

The system operates in two stages:
1. **Stage 1 (Binary Classification):** Predicts whether the machine will fail (`Safe` vs `At Risk`).
2. **Stage 2 (Multi-class Classification):** If the machine is at risk, it diagnoses the specific failure type (TWF, HDF, PWF, OSF, RNF).

## 📁 Project Structure
```text
predictive-maintenance/
│
├── app/
│   └── streamlit_app.py          # Interactive GUI (Industrial Dark Theme)
│
├── notebooks/
│   └── Predictive_Maintenance_Project.ipynb  # Complete ML Pipeline (EDA to Evaluation)
│
├── models/                       # Saved XGBoost models, scalers, and comparison metrics
├── data/                         # Contains predictive_maintenance.csv
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## 🧠 Machine Learning Approach

### 1. Data Preprocessing & Feature Engineering
- **Imbalance Handling:** The dataset was highly imbalanced (3.4% failure rate). We solved this using **SMOTE** combined with class weights.
- **Engineered Features:** Created new critical features based on domain physical laws:
  - `temp_diff = Process Temp - Air Temp` (Crucial for Heat Dissipation Failure)
  - `power = Torque × Rotational Speed` (Crucial for Power Failure)
  - `torque_per_wear = Torque / Tool Wear` (Crucial for Overstrain)

### 2. The Objective Metric: Prioritizing Recall
In Predictive Maintenance, **False Negatives (missing a real failure)** result in catastrophic machine breakdown and high costs, whereas False Positives only result in a routine check. Therefore, we optimized our models primarily for **Recall**, ensuring we catch as many true failures as possible.

---

## 📊 Results & Model Comparison

We rigorously evaluated 6 different algorithms using **GridSearchCV (5-fold CV)**. **XGBoost** outperformed all other models and was selected for the final deployment.

### Stage 1: Binary Classification (Fail / Safe)
| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost (Selected)** | **0.982** | **0.696** | 0.809 | **0.748** | **0.980** |
| Random Forest | 0.973 | 0.571 | 0.824 | 0.675 | 0.976 |
| Decision Tree | 0.963 | 0.471 | 0.824 | 0.599 | 0.895 |
| Logistic Regression | 0.866 | 0.188 | **0.882** | 0.309 | 0.934 |

### Stage 2: Multi-class Classification (Failure Type)
| Model | Accuracy | Precision (W) | Recall (W) | F1-Score (W) |
|-------|----------|---------------|------------|--------------|
| **XGBoost (Selected)** | **0.977** | **0.977** | **0.987** | **0.981** |
| Random Forest | 0.975 | 0.975 | 0.986 | 0.980 |
| Logistic Regression | 0.616 | 0.977 | 0.616 | 0.746 |

> **Note on Metric Discrepancy:** You may notice the Binary F1-Score is 0.748, while the Multi-class Weighted F1-Score is 0.981. This is expected because the multi-class metric is *weighted* by support. The overwhelming majority class ("No Failure", representing 96% of the data) is predicted with near-perfect accuracy, which heavily pulls the weighted average up. The binary F1-score is calculated only on the positive (Failure) class, making it a much stricter metric.

---

## 🖥️ Dashboard & GUI
The project features a fully interactive, industrial-themed dark dashboard built with Streamlit.

🌍 **Live Demo:** [Predictive Maintenance System - Streamlit App](https://predictive-maintenance-system-cmzjnlstjvedkaveex8dy9.streamlit.app/)

<img src="assets/qr_code.png" alt="Scan to open app" width="200"/>

*(Add your screenshots here before uploading to GitHub)*
- `![Dashboard Overview](link_to_image)`
- `![Prediction Interface](link_to_image)`

---

## 🚀 How to Run the Application

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Dashboard:**
   ```bash
   streamlit run app/streamlit_app.py
   ```
3. Open your browser at `http://localhost:8501`.

## ⚠️ Limitations & Future Work
- **Nature of RNF:** "Random Failures" (RNF) are, by definition, stochastic and cannot be reliably predicted by sensor data alone. Performance on this specific class is inherently capped.
- **Synthetic Data:** The AI4I 2020 dataset is synthetically generated to mirror real-world scenarios. While it closely mimics physical laws, testing on real industrial sensor streams would be required for production.
- **Future Work:** Implementing Deep Learning (Neural Networks) and testing real-time stream processing integrations.
