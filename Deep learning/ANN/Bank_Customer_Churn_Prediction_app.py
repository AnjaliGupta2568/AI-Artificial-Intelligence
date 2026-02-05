import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Bank Churn ANN", page_icon="🏦", layout="wide")

st.title("🏦 Bank Customer Churn Prediction (ANN)")
st.write("Upload dataset → Train ANN → Evaluate results")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Churn Dataset (CSV)",
    type=["csv"]
)

epochs = st.sidebar.slider("Epochs", 1, 100, 5)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64])

# --------------------------------------------------
# Main Logic
# --------------------------------------------------
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(dataset.head())

    # -----------------------------
    # Data Preprocessing
    # -----------------------------
    X = dataset.iloc[:, 3:-1].values
    y = dataset.iloc[:, -1].values

    # Label Encoding Gender
    le = LabelEncoder()
    X[:, 2] = le.fit_transform(X[:, 2])

    # One Hot Encoding Geography
    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(), [1])],
        remainder='passthrough'
    )
    X = np.array(ct.fit_transform(X))

    # Feature Scaling
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # -----------------------------
    # Build ANN
    # -----------------------------
    ann = tf.keras.models.Sequential()

    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=5, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=4, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    ann.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # -----------------------------
    # Train Button
    # -----------------------------
    if st.button("🚀 Train ANN Model"):
        with st.spinner("Training model..."):
            history = ann.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=0
            )

        st.success("✅ Model training completed")

        # -----------------------------
        # Prediction & Evaluation
        # -----------------------------
        y_pred = ann.predict(X_test)
        y_pred = (y_pred > 0.5)

        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📌 Confusion Matrix")
            st.write(cm)

        with col2:
            st.subheader("🎯 Accuracy")
            st.metric("Accuracy Score", f"{acc*100:.2f}%")

        # -----------------------------
        # Loss Graph
        # -----------------------------
        st.subheader("📉 Training Loss")
        st.line_chart(history.history['loss'])

else:
    st.info("👈 Upload a CSV file from sidebar to start")
