import streamlit as st
import pandas as pd
from src.preprocessing import preprocess_text
from src.feature_extraction import extract_features
from src.model import load_model

# Load the trained model
model = load_model('/Users/andreaschitos1/Desktop/thesis_git/aphasia-type-classifier/src/aphasia_model.h5')

st.title("Aphasia Type Classifier")

st.write("Upload a .cha file to classify the aphasia type.")

# File uploader
uploaded_file = st.file_uploader("Choose a .cha file", type="cha")

if uploaded_file is not None:
    # Read the uploaded file
    cha_content = uploaded_file.read().decode("utf-8")
    
    # Preprocess the text
    preprocessed_text = preprocess_text(cha_content)
    
    # Extract features
    features = extract_features(preprocessed_text)
    
    # Make prediction
    prediction = model.predict(features)
    predicted_class = "BROCA" if prediction[0][0] > prediction[0][1] else "WERNICKE"
    
    # Display the prediction
    st.write("Predicted Aphasia Type:", predicted_class)

    # Option to download the results
    results_df = pd.DataFrame({'Text': [preprocessed_text], 'Prediction': [predicted_class]})
    st.download_button("Download Results", results_df.to_csv(index=False).encode('utf-8'), "results.csv", "text/csv")