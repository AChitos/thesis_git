import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set page configuration
st.set_page_config(
    page_title="Aphasia Type Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.sub-header {
    font-size: 1.5rem;
    color: #333;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    color: #333;
    border: 1px solid #d0d0d0;
}

.metric-card h4 {
    color: #1f77b4;
    margin-bottom: 0.5rem;
    font-weight: bold;
}

.metric-card p {
    color: #333;
    margin-bottom: 0.3rem;
    line-height: 1.4;
}

.prediction-box {
    background-color: #e8f4fd;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
    margin: 1rem 0;
}

.confidence-high {
    color: #28a745;
    font-weight: bold;
}

.confidence-medium {
    color: #ffc107;
    font-weight: bold;
}

.confidence-low {
    color: #dc3545;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

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

def preprocess_cha_content(content):
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

def extract_features_for_prediction(text, vectorizer):
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

def get_confidence_level(probability):
    """Determine confidence level based on probability."""
    if probability >= 0.8:
        return "High", "confidence-high"
    elif probability >= 0.6:
        return "Medium", "confidence-medium"
    else:
        return "Low", "confidence-low"

def display_model_info():
    """Display information about the model."""
    st.markdown('<div class="sub-header">🤖 Model Information</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>Model Type</h4>
        <p>Deep Neural Network</p>
        <p>• 4 hidden layers</p>
        <p>• Dropout regularization</p>
        <p>• Batch normalization</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>Features</h4>
        <p>TF-IDF Vectorization</p>
        <p>• Max features: 5000</p>
        <p>• Text preprocessing</p>
        <p>• SMOTE balancing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>Classes</h4>
        <p>Binary Classification</p>
        <p>• BROCA Aphasia</p>
        <p>• WERNICKE Aphasia</p>
        </div>
        """, unsafe_allow_html=True)

def display_prediction_results(prediction_proba, preprocessed_text):
    """Display prediction results with confidence scores."""
    # Assuming the model outputs probabilities for [BROCA, WERNICKE]
    broca_prob = prediction_proba[0][0]
    wernicke_prob = prediction_proba[0][1]
    
    # Determine predicted class
    if broca_prob > wernicke_prob:
        predicted_class = "BROCA"
        confidence = broca_prob
    else:
        predicted_class = "WERNICKE"  
        confidence = wernicke_prob
    
    confidence_level, confidence_class = get_confidence_level(confidence)
    
    # Display main prediction
    st.markdown(f"""
    <div class="prediction-box">
    <h3>🎯 Prediction Result</h3>
    <h2 style="color: #1f77b4;">Predicted Aphasia Type: {predicted_class}</h2>
    <p>Confidence: <span class="{confidence_class}">{confidence:.2%} ({confidence_level})</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display probability breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Probability Breakdown")
        prob_data = pd.DataFrame({
            'Aphasia Type': ['BROCA', 'WERNICKE'],
            'Probability': [broca_prob, wernicke_prob]
        })
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(prob_data['Aphasia Type'], prob_data['Probability'], 
                     color=['#ff7f0e' if predicted_class == 'BROCA' else '#1f77b4',
                            '#1f77b4' if predicted_class == 'WERNICKE' else '#ff7f0e'])
        
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, prob in zip(bars, [broca_prob, wernicke_prob]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{prob:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### 📈 Confidence Meter")
        
        # Create a confidence gauge
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Color segments
        colors = ['red', 'orange', 'green']
        segments = [0.6, 0.8, 1.0]
        
        for i, (color, segment) in enumerate(zip(colors, segments)):
            start = 0 if i == 0 else segments[i-1]
            mask = (confidence >= start) & (confidence <= segment)
            ax.fill_between(theta, 0, r, where=(theta <= confidence * np.pi), 
                          color=color, alpha=0.3)
        
        # Add needle
        needle_angle = confidence * np.pi
        ax.plot([needle_angle, needle_angle], [0, 1], 'k-', linewidth=3)
        
        ax.set_ylim(0, 1)
        ax.set_xlim(0, np.pi)
        ax.set_title(f'Confidence: {confidence:.1%}')
        ax.set_xticks([0, np.pi/2, np.pi])
        ax.set_xticklabels(['0%', '50%', '100%'])
        
        plt.tight_layout()
        st.pyplot(fig)

def main():
    # Header
    st.markdown('<div class="main-header">🧠 Aphasia Type Classifier</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    model, vectorizer = load_models()
    
    if model is None or vectorizer is None:
        st.error("Failed to load the required models. Please check the model files.")
        st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("## 📋 Instructions")
        st.markdown("""
        1. Upload a .cha file containing aphasia speech data
        2. The system will preprocess the text automatically
        3. Get instant predictions with confidence scores
        4. Download results for further analysis
        """)
        
        st.markdown("---")
        st.markdown("## ℹ️ About Aphasia")
        st.markdown("""
        **Broca's Aphasia:**
        - Difficulty with speech production
        - Limited vocabulary
        - Grammar issues
        
        **Wernicke's Aphasia:**
        - Fluent but meaningless speech
        - Word-finding difficulties
        - Comprehension problems
        """)
    
    # Model information
    display_model_info()
    
    st.markdown("---")
    st.markdown('<div class="sub-header">📁 Upload Your Data</div>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a .cha file", 
        type="cha",
        help="Upload a CHAT format file containing aphasia speech data"
    )
    
    if uploaded_file is not None:
        # Display file information
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Read and preprocess the file
        try:
            # Read file content
            cha_content = uploaded_file.read().decode("utf-8")
            
            # Display raw content preview
            with st.expander("🔍 View Raw File Content"):
                st.text_area("Raw content:", cha_content, height=200)
            
            # Preprocess the text
            preprocessed_text = preprocess_cha_content(cha_content)
            
            if not preprocessed_text.strip():
                st.warning("⚠️ No participant speech found in the file. Please check the file format.")
                st.stop()
            
            # Display preprocessed content
            with st.expander("🔧 View Preprocessed Content"):
                st.text_area("Preprocessed text:", preprocessed_text, height=150)
            
            # Extract features
            features = extract_features_for_prediction(preprocessed_text, vectorizer)
            
            # Make prediction
            with st.spinner("🔮 Making prediction..."):
                prediction_proba = model.predict(features)
            
            # Display results
            display_prediction_results(prediction_proba, preprocessed_text)
            
            # Results summary for download
            st.markdown("---")
            st.markdown('<div class="sub-header">💾 Download Results</div>', unsafe_allow_html=True)
            
            # Prepare results data
            broca_prob = prediction_proba[0][0]
            wernicke_prob = prediction_proba[0][1]
            predicted_class = "BROCA" if broca_prob > wernicke_prob else "WERNICKE"
            confidence = max(broca_prob, wernicke_prob)
            
            results_df = pd.DataFrame({
                'Filename': [uploaded_file.name],
                'Predicted_Class': [predicted_class],
                'Confidence': [f"{confidence:.4f}"],
                'BROCA_Probability': [f"{broca_prob:.4f}"],
                'WERNICKE_Probability': [f"{wernicke_prob:.4f}"],
                'Preprocessed_Text': [preprocessed_text],
                'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            })
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv_data,
                    file_name=f"aphasia_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_data = results_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📋 Download as JSON",
                    data=json_data,
                    file_name=f"aphasia_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            # Display results table
            st.markdown("### 📊 Results Summary")
            st.dataframe(results_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.error("Please check that your file is in the correct CHAT format.")
    
    else:
        # Example section
        st.markdown("---")
        st.markdown('<div class="sub-header">📝 Example Usage</div>', unsafe_allow_html=True)
        
        example_content = """@Begin
@Languages: eng
@Participants: PAR Participant, INV Investigator
*PAR: the boy is climbing the tree
*INV: what else do you see?
*PAR: bird flying away
*PAR: the cat underneath
@End"""
        
        st.markdown("**Example .cha file content:**")
        st.code(example_content, language="text")
        
        st.markdown("""
        **The system will:**
        1. Extract only participant speech lines (*PAR:)
        2. Remove punctuation and normalize text
        3. Convert to features using TF-IDF vectorization
        4. Make predictions using the neural network
        """)

if __name__ == "__main__":
    main()
