import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load('models/iris_classifier_model.pkl')
scaler = joblib.load('models/iris_scaler.pkl')

st.title("🌸 Iris Flower Species Classifier")
st.markdown("Enter the flower measurements below to predict the species.")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.1, 0.1)
with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 3.8, 0.1)
    petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_scaled = scaler.transform(input_data)

if st.button("Predict Species"):
    prediction = model.predict(input_scaled)[0].capitalize()
    probabilities = model.predict_proba(input_scaled)[0]
    
    st.success(f"**Predicted Species: {prediction}**")
    
    prob_df = pd.DataFrame({
        'Species': [s.capitalize() for s in model.classes_],
        'Probability': probabilities
    }).sort_values('Probability', ascending=False)
    
    st.bar_chart(prob_df.set_index('Species'))
    st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}))
