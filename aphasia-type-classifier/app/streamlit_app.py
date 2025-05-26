import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import re
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

@st.cache_resource
def load_models():
    """Load the trained neural network model and vectorizer."""
    try:
        # Load the neural network model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'aphasia_model.h5')
        model = load_model(model_path)
        
        # Load the vectorizer
        vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'vectorizer.pkl')
        vectorizer = joblib.load(vectorizer_path)
        
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def preprocess_text(content):
    """
    Preprocess the content of a .cha file for prediction.
    
    Args:
        content (str): Raw content of the .cha file
        
    Returns:
        str: Preprocessed text ready for feature extraction
    """
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Remove lines starting with '@', '*INV:', and '%wor:'
        if line.startswith('@') or line.startswith('*INV:') or line.startswith('%wor:'):
            continue
        
        # Keep only lines starting with '*PAR:'
        if line.startswith('*PAR:'):
            # Normalize and clean the text
            text = line[5:].strip()  # Remove '*PAR:' prefix
            text = text.lower()  # Lowercase
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            cleaned_lines.append(text)
    
    # Join all cleaned lines into a single text
    return ' '.join(cleaned_lines)

def extract_features(text, vectorizer):
    """
    Extract features from preprocessed text using the trained vectorizer.
    
    Args:
        text (str): Preprocessed text
        vectorizer: Trained TF-IDF vectorizer
        
    Returns:
        numpy.ndarray: Feature vector for prediction
    """
    # Transform text using the trained vectorizer
    features = vectorizer.transform([text])
    return features.toarray()

# Load the trained model and vectorizer
model, vectorizer = load_models()

if model is None or vectorizer is None:
    st.error("Failed to load the required models. Please check the model files.")
    st.stop()

st.title("Aphasia Type Classifier")

st.write("Upload a .cha file to classify the aphasia type.")

# File uploader
uploaded_file = st.file_uploader("Choose a .cha file", type="cha")

if uploaded_file is not None:
    # Read the uploaded file
    cha_content = uploaded_file.read().decode("utf-8")
    
    # Preprocess the text
    preprocessed_text = preprocess_text(cha_content)
    
    if not preprocessed_text.strip():
        st.warning("No participant speech found in the file. Please check the file format.")
    else:
        # Extract features
        features = extract_features(preprocessed_text, vectorizer)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Get probabilities
        broca_prob = prediction[0][0]
        wernicke_prob = prediction[0][1]
        
        # Determine predicted class
        predicted_class = "BROCA" if broca_prob > wernicke_prob else "WERNICKE"
        confidence = max(broca_prob, wernicke_prob)
        
        # Display the prediction
        st.success(f"**Predicted Aphasia Type: {predicted_class}**")
        st.write(f"**Confidence:** {confidence:.2%}")
        
        # Show probability breakdown
        st.write("**Probability Breakdown:**")
        prob_df = pd.DataFrame({
            'Aphasia Type': ['BROCA', 'WERNICKE'],
            'Probability': [f"{broca_prob:.4f}", f"{wernicke_prob:.4f}"]
        })
        st.dataframe(prob_df, use_container_width=True)
        
        # Show preprocessed text
        with st.expander("View Preprocessed Text"):
            st.text_area("Preprocessed content:", preprocessed_text, height=150)

        # Option to download the results
        results_df = pd.DataFrame({
            'Filename': [uploaded_file.name],
            'Predicted_Class': [predicted_class],
            'Confidence': [f"{confidence:.4f}"],
            'BROCA_Probability': [f"{broca_prob:.4f}"],
            'WERNICKE_Probability': [f"{wernicke_prob:.4f}"],
            'Preprocessed_Text': [preprocessed_text]
        })
        
        st.download_button(
            "Download Results", 
            results_df.to_csv(index=False).encode('utf-8'), 
            f"aphasia_prediction_{uploaded_file.name}.csv", 
            "text/csv"
        )