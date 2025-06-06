import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, ConfusionMatrixDisplay
)
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Heart Attack Prediction App", layout="wide")
st.title("💓 Heart Attack Prediction App")

# User Inputs
st.sidebar.header("🧑 User Information")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=25)
email = st.sidebar.text_input("Email")
motivation = st.sidebar.text_area("What's your motivation to check your heart health?")

model_choice = st.sidebar.radio("Choose Model", ["Naive Bayes", "AdaBoost"])

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📤 Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Load data
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(10))

    st.markdown("### ❗ Missing Values per Feature")
    st.write(df.isnull().sum())

    st.markdown("### ⚖️ Class Distribution")
    if 'heart_attack' in df.columns:
        st.bar_chart(df['heart_attack'].value_counts())
    else:
        st.warning("Target column `heart_attack` not found.")
        st.stop()

    st.markdown("### 📊 Data Cleaning & Preprocessing")

    # Handle numeric missing values
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Drop remaining rows with missing values
    df.dropna(inplace=True)

    # Outlier removal using z-score
    z_scores = stats.zscore(df.select_dtypes(include=['number']))
    outliers = (abs(z_scores) > 2).any(axis=1)
    df_cleaned = df[~outliers]

    # Encode categorical columns
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df_cleaned[col] = label_encoders[col].fit_transform(df_cleaned[col])

    # Split X/y
    X = df_cleaned.drop(columns=["heart_attack"])
    y = df_cleaned["heart_attack"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=min(5, X_train.shape[1]))
    X_selected = selector.fit_transform(X_train, y_train)
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices]
    st.success(f"Selected Features: {list(selected_features)}")

    # Train model
    model = None
    if model_choice == "Naive Bayes":
        model = GaussianNB()
    else:
        model = AdaBoostClassifier(n_estimators=50, random_state=42)

    model.fit(X_train[selected_features], y_train)
    y_pred = model.predict(X_test[selected_features])
    y_proba = model.predict_proba(X_test[selected_features])[:, 1]

    # Evaluation Metrics
    st.subheader(f"📈 {model_choice} Model Results")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"**Precision:** {precision_score(y_test, y_pred):.4f}")
    st.write(f"**Recall:** {recall_score(y_test, y_pred):.4f}")
    st.write(f"**F1 Score:** {f1_score(y_test, y_pred):.4f}")

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    st.subheader("🧪 ROC Curve")
    fig_roc, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_choice}')
    ax.legend()
    st.pyplot(fig_roc)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("🔢 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # Predict for user input (simplified example)
    st.header("🔮 Final Prediction")
    input_data = {}
    for feature in selected_features:
        val = st.number_input(f"Enter your value for '{feature}'", value=0)
        input_data[feature] = val

    if st.button("Predict Heart Attack Risk"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        result = "🚨 At Risk of Heart Attack" if prediction == 1 else "✅ Not at Risk"
        st.subheader(f"Result for {name} ({age} years old)")
        st.write(f"📧 Email: {email}")
        st.write(f"💬 Motivation: {motivation}")
        st.success(f"**Prediction:** {result}")
else:
    st.info("Please upload a dataset to continue.")
