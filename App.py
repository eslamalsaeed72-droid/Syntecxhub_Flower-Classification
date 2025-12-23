import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler (adjust path if in models folder)
model = joblib.load('models/iris_classifier_model.pkl')  # أو 'iris_classifier_model.pkl' لو في الroot
scaler = joblib.load('models/iris_scaler.pkl')

# Page config
st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="centered")

# Custom CSS for better styling
st.markdown("""
<style>
    .main {background-color: #f9f9fb;}
    .stSlider > div > div > div > div {background-color: #e6e6fa;}
    .species-img {border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    h1 {color: #4b0082; text-align: center;}
    .stSuccess {font-size: 1.5em; text-align: center;}
</style>
""", unsafe_allow_html=True)

# Header with images
st.title("🌸 Iris Flower Species Classifier 🌸")
st.markdown("### Enter the flower measurements to predict its species with high accuracy!")

col1, col2, col3 = st.columns(3)
with col1:
    st.image("https://thumbs.dreamstime.com/b/closeup-single-iris-setosa-flower-pretty-mauve-garden-228835113.jpg", caption="Iris Setosa", width=200)
with col2:
    st.image("https://thumbs.dreamstime.com/b/blue-flowers-iris-versicolor-beautifully-blooming-garden-172555133.jpg", caption="Iris Versicolor", width=200)
with col3:
    st.image("https://thumbs.dreamstime.com/b/virginia-iris-flower-virginica-168902047.jpg", caption="Iris Virginica", width=200)

st.markdown("---")

# Input sliders
st.subheader("Flower Measurements (in cm)")
col_left, col_right = st.columns(2)
with col_left:
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8, 0.1, help="Length of the sepal")
    sepal_width  = st.slider("Sepal Width", 2.0, 4.5, 3.1, 0.1, help="Width of the sepal")
with col_right:
    petal_length = st.slider("Petal Length", 1.0, 7.0, 3.8, 0.1, help="Length of the petal")
    petal_width  = st.slider("Petal Width", 0.1, 2.5, 1.2, 0.1, help="Width of the petal")

# Prepare input
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("🌟 Predict Species 🌟", type="primary"):
    prediction = model.predict(input_scaled)[0].capitalize()
    probabilities = model.predict_proba(input_scaled)[0]
    
    prob_df = pd.DataFrame({
        'Species': [s.capitalize() for s in model.classes_],
        'Probability': probabilities
    }).sort_values('Probability', ascending=True)
    
    # Display result
    st.success(f"**Predicted Species: {prediction}**")
    
    # Show image of predicted species
    species_images = {
        "Setosa": "https://thumbs.dreamstime.com/b/closeup-single-iris-setosa-flower-pretty-mauve-garden-228835113.jpg",
        "Versicolor": "https://thumbs.dreamstime.com/b/blue-flowers-iris-versicolor-beautifully-blooming-garden-172555133.jpg",
        "Virginica": "https://thumbs.dreamstime.com/b/virginia-iris-flower-virginica-168902047.jpg"
    }
    st.image(species_images[prediction], caption=f"{prediction} Flower", width=400)
    
    # Bar chart for probabilities
    st.subheader("Prediction Confidence")
    st.bar_chart(prob_df.set_index('Species')['Probability'].apply(lambda x: x*100))
    
    # Table with percentages
    prob_df['Probability (%)'] = (prob_df['Probability'] * 100).round(2)
    st.dataframe(prob_df[['Species', 'Probability (%)']], use_container_width=True)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Model: Decision Tree (96.67% Accuracy) | Dataset: Iris (scikit-learn)")
