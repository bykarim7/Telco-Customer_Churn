# 📉 Telco Customer Churn Prediction

A Machine Learning project that predicts whether a telecommunications customer will churn (cancel their service) based on demographic, account, and service usage data. The project demonstrates an end-to-end machine learning pipeline, covering exploratory data analysis, data cleaning, class imbalance handling, cross-validation, and model serialization.

---

## 📌 Project Overview

Customer attrition is a critical challenge for subscription-based telecommunication companies. Identifying churn-prone customers early allows businesses to implement targeted retention strategies and reduce revenue loss.

In this project, binary classification models—specifically **Random Forest** and **Support Vector Classifier (SVC)**—were built and evaluated to accurately predict customer churn (`Churn = Yes/No`).

---

## 📂 Dataset

The project utilizes the **Telco Customer Churn Dataset** (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), which contains **7,043 customer records** and **21 features**.

### Features Overview

| Category | Features | Description |
|---|---|---|
| **Customer Metadata** | `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Demographic attributes (Note: `customerID` was removed prior to modeling). |
| **Services Signed Up** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Telecom services and add-ons subscribed to by the customer. |
| **Account Information** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` | Contract duration, billing settings, tenure duration, and payment metrics. |

### Target Variable

- **No** → Customer Retained (Stays with the service)
- **Yes** → Customer Churned (Leaves the service)

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (`SMOTE`)
- Pickle

---

# 📊 Project Workflow

## 1. Initial Data Inspection & Cleaning

- Loaded dataset and inspected initial shape (7,043 × 21).
- Removed non-informative identifier (`customerID`).
- Identified 11 records containing blank space strings (`' '`) in the `TotalCharges` column, which corresponded to new customers (`tenure = 0`).
- Handled missing values by replacing blank strings with `'0.0'` and converting `TotalCharges` to `float64`.
- Examined unique categorical levels and analyzed feature combinations for duplicate entries.

---

## 2. Feature Preprocessing & Encoding

- Converted binary and multi-class categorical features into numerical formats suitable for machine learning algorithms.
- Normalized continuous variables (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler` and `MinMaxScaler` to equalize feature scales.

---

## 3. Class Imbalance Handling (SMOTE)

Because churned customers represent a minority class, the **Synthetic Minority Over-sampling Technique (SMOTE)** was applied to balance the target class distribution prior to final model training.

---

## 4. Train-Test Split & Validation Strategy

- Split data using a stratified train-test strategy to maintain target class distributions across sets.
- Applied cross-validation techniques (`cross_val_score`, `cross_val_predict`) to evaluate model generalization and prevent overfitting.

---

# 🤖 Models Used

## Random Forest Classifier

An ensemble tree-based algorithm trained to capture non-linear relationships, feature interactions, and complex decision boundaries.

---

## Support Vector Machine (SVC)

A kernel-based classifier evaluated on normalized and scaled features to separate churned and retained customer groups effectively.

---

# 📈 Model Evaluation

Models were evaluated across multiple performance metrics to balance minimizing false negatives (missing actual churners) and false positives:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

---

# 💾 Model Serialization

Trained model artifacts and scaling transformers were serialized using Python's `pickle` library to facilitate deployment and inference on new customer data.

---

# 📁 Project Structure

```
Telco-Customer-Churn-Prediction/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── telco_churn_analysis.ipynb
├── model.pkl
├── scaler.pkl
├── README.md
└── requirements.txt
```

---

# 🚀 How to Run

## Clone the repository

```bash
git clone https://github.com/your-username/Telco-Customer-Churn-Prediction.git
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook and run all cells.

---

# 📌 Key Machine Learning Concepts Demonstrated

- Data Cleaning & Type Conversion
- Feature Engineering & Scaling
- Handling Class Imbalance (SMOTE)
- Stratified Train/Test Split
- Cross-Validation Strategies
- Random Forest & Support Vector Classifiers
- Comprehensive Classification Metrics Evaluation
- Model Serialization (`pickle`)

---

# 📚 Future Improvements

- Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`.
- Incorporate gradient boosting models (XGBoost, LightGBM, CatBoost).
- Develop an interactive web interface using Streamlit for dynamic churn risk scoring.
- Containerize the inference pipeline using Docker and build a FastAPI web service.

---

# 👨‍💻 Author

**Karim Mohamed**  
Machine Learning & AI Student

---

## ⭐ If you found this project useful, consider giving it a star!
