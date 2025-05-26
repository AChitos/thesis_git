#!/usr/bin/env python3
"""
Test script to verify that the neural network model and vectorizer work correctly
"""

import os
import sys
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import re

def preprocess_test_text(content):
    """Preprocess test content similar to the Streamlit app."""
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if line.startswith('@') or line.startswith('*INV:') or line.startswith('%wor:'):
            continue
        
        if line.startswith('*PAR:'):
            text = line[5:].strip()
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            cleaned_lines.append(text)
    
    return ' '.join(cleaned_lines)

def test_model():
    """Test the model with a sample input."""
    print("Testing Aphasia Type Classifier Model...")
    print("=" * 50)
    
    # Load models
    try:
        model_path = "src/aphasia_model.h5"
        vectorizer_path = "src/vectorizer.pkl"
        
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        
        print(f"Loading vectorizer from: {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
        
        print("✅ Models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        return
    
    # Test with sample data
    print("\nTesting with sample .cha content...")
    
    sample_content = """@Begin
@Languages: eng
@Participants: PAR Participant, INV Investigator
*PAR: the boy is climbing the tree
*INV: what else do you see?
*PAR: bird flying away
*PAR: the cat underneath looking up
*PAR: very high tree with green leaves
@End"""
    
    print("Sample content:")
    print(sample_content)
    
    # Preprocess
    preprocessed_text = preprocess_test_text(sample_content)
    print(f"\nPreprocessed text: '{preprocessed_text}'")
    
    if not preprocessed_text.strip():
        print("❌ No text extracted from sample!")
        return
    
    # Extract features
    features = vectorizer.transform([preprocessed_text]).toarray()
    print(f"Feature vector shape: {features.shape}")
    
    # Make prediction
    prediction = model.predict(features)
    print(f"Raw prediction output: {prediction}")
    
    # Interpret results
    broca_prob = prediction[0][0]
    wernicke_prob = prediction[0][1]
    
    predicted_class = "BROCA" if broca_prob > wernicke_prob else "WERNICKE"
    confidence = max(broca_prob, wernicke_prob)
    
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    print(f"BROCA Probability: {broca_prob:.4f}")
    print(f"WERNICKE Probability: {wernicke_prob:.4f}")
    print("=" * 50)
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    # Change to the project directory
    os.chdir("/Users/andreaschitos1/Desktop/thesis_git/aphasia-type-classifier")
    test_model()
