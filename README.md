# Heart-Attack-predictor
Heart Attack prediction using machine learning models
# 💓 Heart Attack Prediction Web App

This is a **Streamlit-based machine learning web application** for predicting the risk of a heart attack using user health data and machine learning models. The app supports **Naive Bayes** and **AdaBoost** classifiers, provides interactive visualizations, and enables users to upload their own dataset (CSV or Excel).

---

## 🚀 Live Demo
> _Coming soon (you can deploy on [Streamlit Community Cloud](https://share.streamlit.io) or [Render](https://render.com))_

---

## 📌 Features

- Upload your custom dataset (CSV or Excel).
- Automatic data preprocessing:
  - Handling missing values
  - Removing outliers using Z-score
  - Encoding categorical features
  - Feature selection using ANOVA (SelectKBest)
- Choose between:
  - **Naive Bayes**
  - **AdaBoost Classifier**
- Get evaluation metrics:
  - Accuracy, Precision, Recall, F1 Score
  - Confusion Matrix
  - ROC Curve and AUC score
- Enter your **personal information** (name, age, email, motivation).
- Get a personalized prediction: **Heart attack risk or not**.
- Clean, interactive, and responsive UI powered by Streamlit.

---

## 📂 Dataset Requirements

Your dataset must:

- Contain a target column named `heart_attack`.
- Have features like `diabetes`, `hypertension`, `obesity`, `smoking_status`, etc.
- Include both numeric and categorical features.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/heart-attack-prediction-app.git
cd heart-attack-prediction-app
