#!/usr/bin/env python3
"""
Test the Streamlit app functionality with real data files
"""

import os
import sys
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import re

def preprocess_cha_file(filepath):
    """Load and preprocess a real .cha file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    cleaned_lines = []
    actual_label = None
    
    # Extract actual label from first line
    if lines and not lines[0].startswith('@'):
        actual_label = lines[0].strip()
    
    for line in lines:
        if line.startswith('@') or line.startswith('*INV:') or line.startswith('%wor:'):
            continue
        
        if line.startswith('*PAR:'):
            text = line[5:].strip()
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            if text.strip():
                cleaned_lines.append(text)
    
    return ' '.join(cleaned_lines), actual_label

def test_with_real_data():
    """Test the model with real data files."""
    print("🧠 Testing Aphasia Classifier with Real Data")
    print("=" * 60)
    
    # Load models
    try:
        model_path = "aphasia-type-classifier/src/aphasia_model.h5"
        vectorizer_path = "aphasia-type-classifier/src/vectorizer.pkl"
        
        model = load_model(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("✅ Models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        return
    
    # Test with several real files
    test_files = [
        "aphasia-type-classifier/data/02story.cha",
        "aphasia-type-classifier/data/05free.cha", 
        "aphasia-type-classifier/data/11story.cha",
        "aphasia-type-classifier/data/26free.cha"
    ]
    
    results = []
    
    for filepath in test_files:
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}")
            continue
            
        print(f"\n📄 Testing: {os.path.basename(filepath)}")
        print("-" * 40)
        
        try:
            # Preprocess file
            preprocessed_text, actual_label = preprocess_cha_file(filepath)
            
            if not preprocessed_text.strip():
                print("❌ No participant speech found!")
                continue
            
            print(f"Actual Label: {actual_label}")
            print(f"Text Preview: {preprocessed_text[:100]}...")
            
            # Extract features and predict
            features = vectorizer.transform([preprocessed_text]).toarray()
            prediction = model.predict(features, verbose=0)
            
            broca_prob = prediction[0][0]
            wernicke_prob = prediction[0][1]
            predicted_class = "BROCA" if broca_prob > wernicke_prob else "WERNICKE"
            confidence = max(broca_prob, wernicke_prob)
            
            # Check if prediction matches actual
            correct = predicted_class == actual_label
            
            print(f"Predicted: {predicted_class} (Confidence: {confidence:.2%})")
            print(f"Result: {'✅ CORRECT' if correct else '❌ INCORRECT'}")
            
            results.append({
                'file': os.path.basename(filepath),
                'actual': actual_label,
                'predicted': predicted_class,
                'confidence': confidence,
                'correct': correct,
                'broca_prob': broca_prob,
                'wernicke_prob': wernicke_prob
            })
            
        except Exception as e:
            print(f"❌ Error processing {filepath}: {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY RESULTS")
    print("=" * 60)
    
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        total_count = len(results)
        accuracy = correct_count / total_count
        
        print(f"Files Tested: {total_count}")
        print(f"Correct Predictions: {correct_count}")
        print(f"Accuracy: {accuracy:.2%}")
        
        print("\nDetailed Results:")
        print("-" * 60)
        for r in results:
            status = "✅" if r['correct'] else "❌"
            print(f"{status} {r['file']:<20} | Actual: {r['actual']:<8} | Predicted: {r['predicted']:<8} | Conf: {r['confidence']:.2%}")
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    # Change to the workspace directory
    os.chdir("/Users/andreaschitos1/Desktop/thesis_git")
    test_with_real_data()
