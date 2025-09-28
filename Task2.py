import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import os
import sys

def app():
    st.title("🔮 Customer Churn Prediction")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ File uploaded successfully!")
    else:
        st.info("⚠️ No file uploaded. Using sample dataset instead.")
        data = {
            'Gender': ['Male','Female','Female','Male','Male','Female','Female','Male','Female','Male'],
            'Age': [25, 30, 45, 35, 40, 23, 50, 44, 36, 29],
            'Tenure': [1, 3, 5, 2, 8, 1, 10, 6, 7, 2],
            'Balance': [20000, 50000, 60000, 30000, 80000, 10000, 120000, 90000, 40000, 15000],
            'NumOfProducts': [1, 2, 1, 3, 2, 1, 2, 2, 1, 3],
            'HasCrCard': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
            'IsActiveMember': [1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
            'EstimatedSalary': [50000, 60000, 52000, 70000, 80000, 45000, 90000, 75000, 62000, 58000],
            'Exited': [0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
        }
        df = pd.DataFrame(data)
        
    if 'Gender' in df.columns:
        encoder = LabelEncoder()
        df['Gender'] = encoder.fit_transform(df['Gender'])

    if 'Exited' not in df.columns:
        st.error("❌ Dataset must contain an 'Exited' column (target).")
        return

    X = df.drop('Exited', axis=1)
    y = df['Exited']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    if len(y_train.value_counts()) > 1 and min(y_train.value_counts()) > 1:
        k_neighbors = min(5, min(y_train.value_counts()) - 1)
        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=300,     
        max_depth=10,          
        max_features="sqrt",   
        min_samples_split=5,   
        min_samples_leaf=2,    
        random_state=42,
        class_weight="balanced",
        n_jobs=-1              
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    cv_scores = None
    if df.shape[0] < 5000:
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)

    tab1, tab2 = st.tabs(["Model Evaluation", "New Prediction"])

    with tab1:
        st.header("Model Performance")
        st.write(f"**Accuracy (holdout):** {acc:.2f}")
        if cv_scores is not None:
            st.write(f"**5-Fold Cross-Validated Accuracy:** {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
        st.dataframe(pd.DataFrame(report).transpose())

    with tab2:
        st.header("Predict New Customer")

        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 70, 30)
        tenure = st.number_input("Tenure (years)", 0, 20, 5)
        balance = st.number_input("Balance", 0, 200000, 50000)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_card = st.radio("Has Credit Card?", [0, 1])
        active_member = st.radio("Is Active Member?", [0, 1])
        salary = st.number_input("Estimated Salary", 10000, 200000, 60000)

        new_customer = np.array([[ 
            1 if gender == "Male" else 0, age, tenure, balance,
            num_products, has_card, active_member, salary
        ]])

        new_customer_scaled = scaler.transform(new_customer)
        prediction = model.predict(new_customer_scaled)
        proba = model.predict_proba(new_customer_scaled)[0][1]

        if prediction[0] == 1:
            st.error("⚠️ This customer is likely to CHURN.")
        else:
            st.success("✅ This customer is NOT likely to churn.")

        st.write(f"**Churn Probability:** {proba:.2%}")

if __name__ == "__main__":
    if os.environ.get("RUNNING_STREAMLIT") != "true":
        os.environ["RUNNING_STREAMLIT"] = "true"
        os.system(f"streamlit run {sys.argv[0]}")
    else:
        app()

