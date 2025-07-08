# 🌟 Customer Churn Prediction using Artificial Neural Networks

<p align="center">
  <img src="https://img.shields.io/badge/Project%20Status-Completed-green" />
  <img src="https://img.shields.io/badge/Model-ANN-blueviolet" />
  <img src="https://img.shields.io/badge/Deployment-Streamlit-orange" />
</p>

---

## 📌 **Table of Contents**

1. [Introduction](#introduction)
2. [Problem Definition](#problem-definition)
3. [Dataset & Preprocessing](#dataset--preprocessing)
4. [Methodology](#methodology)
5. [Implementation Pipeline](#implementation-pipeline)
6. [Streamlit Application](#streamlit-application)
7. [Conclusion](#conclusion)
8. [Project Structure](#project-structure)

---

## 📝 **Introduction**

This project focuses on developing an **Artificial Neural Network (ANN)** to predict **customer churn** in the banking sector. Customer churn is when customers stop using a service, leading to potential revenue loss. Accurate churn prediction empowers banks to implement **targeted retention strategies**, maximizing customer lifetime value.

---

## ❓ **Problem Definition**

- **Objective:** Predict which bank customers are likely to churn (close their accounts).
- **Type:** Binary classification problem.
- **Target:** `Exited` column (1 = Churn, 0 = Not Churn).

### 🔹 **Churn Examples Across Industries**
- **Telecom:** Cancelling mobile/internet plan.
- **Streaming:** Cancelling Netflix/Spotify subscription.
- **E-commerce:** Customers stop purchasing.
- **Banking:** Customer closes bank account.

This project focuses on **banking churn prediction**.

---

## 📂 **Dataset & Preprocessing**

### 🔸 **Dataset:** `Churn_Modelling.csv`

| Feature | Description |
|---|---|
| CreditScore | Customer's credit score |
| Geography | Country (France, Spain, Germany) |
| Gender | Male/Female |
| Age | Customer's age |
| Tenure | Years with bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Has credit card (1/0) |
| IsActiveMember | Active member (1/0) |
| EstimatedSalary | Customer's estimated salary |
| Exited | **Target**: Churn status (1/0) |

**Rows:** 10,000 customers

---

### 🔸 **Preprocessing Steps**

✅ Dropped irrelevant columns: `RowNumber`, `CustomerId`, `Surname`  
✅ **Label Encoding:** Gender  
✅ **One-Hot Encoding:** Geography  
✅ **Feature Scaling:** StandardScaler  
✅ **Train-Test Split:** 80%-20%

All preprocessing encoders and scalers were saved for consistent deployment.

---

## 🧠 **Methodology**

### 🔹 **Model: Artificial Neural Network (ANN)**

| Layer | Details |
|---|---|
| **Input Layer** | 12 features |
| **Hidden Layer 1** | 64 neurons, ReLU activation |
| **Hidden Layer 2** | 32 neurons, ReLU activation |
| **Output Layer** | 1 neuron, Sigmoid activation |

✅ **Optimizer:** Adam (lr=0.01)  
✅ **Loss Function:** Binary Crossentropy  
✅ **Metrics:** Accuracy  
✅ **Callbacks:** EarlyStopping, TensorBoard  
✅ **Training:** Up to 100 epochs with early stopping

📊 **Total Parameters:** 2945

---

## ⚙️ **Implementation Pipeline**

1. **Data Collection:** Customer attributes collected.
2. **Preprocessing:** Gender encoding, geography encoding, scaling.
3. **Prediction:** Model outputs churn probability.
4. **Interpretation:** Probability threshold (e.g., 0.5) classifies customer as churn or not.

---

## 🚀 **Streamlit Application**

A **user-friendly web app** was built for deployment:

🔹 **Features:**
- Collects user input via interactive widgets.
- Applies saved encoders and scaler.
- Predicts churn probability instantly.
- Displays clear, actionable results.

🔹 **Usage:**  
Run the app locally with:

```bash
streamlit run app.py
```

## ``` 📦 Customer Churn Prediction/
├── data/
│ └── Churn_Modelling.csv
├── models/
│ ├── churn_model.h5
│ ├── label_encoder_gender.pkl
│ ├── one_hot_encoder_geo.pkl
│ └── scaler.pkl
├── notebooks/
│ └── experiment.ipynb
├── app.py
├── prediction.ipynb
└── README.md
```
