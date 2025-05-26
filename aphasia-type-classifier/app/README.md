# Aphasia Type Classifier - Streamlit App

This Streamlit application provides an interactive interface for classifying aphasia types using a trained neural network model.

## Features

### 🎯 Main Features
- **File Upload**: Support for .cha (CHAT format) files
- **Real-time Prediction**: Instant classification using neural network
- **Confidence Scores**: Detailed probability breakdown for both classes
- **Interactive Visualizations**: Charts showing prediction confidence
- **Results Export**: Download predictions as CSV or JSON

### 🧠 Model Information
- **Architecture**: Deep Neural Network with 4 hidden layers
- **Features**: TF-IDF vectorization (max 5000 features)
- **Classes**: BROCA and WERNICKE aphasia types
- **Training**: SMOTE balancing with class weights

## How to Use

### 1. Running the Application

#### Enhanced Version (Recommended)
```bash
cd aphasia-type-classifier
python3 -m streamlit run app/enhanced_streamlit_app.py
```

#### Simple Version
```bash
cd aphasia-type-classifier
python3 -m streamlit run app/streamlit_app.py
```

### 2. Using the Interface

1. **Upload File**: Click "Choose a .cha file" and select your CHAT format file
2. **View Processing**: See raw and preprocessed content in expandable sections
3. **Get Predictions**: View the predicted aphasia type with confidence scores
4. **Download Results**: Export predictions in CSV or JSON format

### 3. File Format Requirements

Your .cha file should follow the CHAT format:
```
@Begin
@Languages: eng
@Participants: PAR Participant, INV Investigator
*PAR: participant speech here
*INV: investigator speech here
*PAR: more participant speech
@End
```

**Important**: Only lines starting with `*PAR:` (participant speech) are used for classification.

## Model Files Required

The application requires these files in the `src/` directory:
- `aphasia_model.h5` - Trained neural network model
- `vectorizer.pkl` - TF-IDF vectorizer used during training

## Example Usage

### Sample Input File
```
@Begin
@Languages: eng
@Participants: PAR Participant, INV Investigator
*PAR: the boy is climbing the tree
*INV: what else do you see?
*PAR: bird flying away
*PAR: the cat underneath
@End
```

### Expected Output
- **Predicted Class**: BROCA or WERNICKE
- **Confidence**: Percentage confidence in prediction
- **Probability Breakdown**: Individual probabilities for each class
- **Downloadable Results**: Structured data with all prediction details

## Technical Details

### Preprocessing Pipeline
1. Extract lines starting with `*PAR:`
2. Remove punctuation and normalize text
3. Convert to lowercase
4. Join all participant utterances

### Feature Extraction
- Uses pre-trained TF-IDF vectorizer
- 5000 maximum features
- Transforms text to numerical features

### Prediction
- Neural network processes feature vector
- Outputs probabilities for both classes
- Applies softmax activation for probability distribution

## Troubleshooting

### Common Issues

1. **"No participant speech found"**
   - Ensure your file contains lines starting with `*PAR:`
   - Check file encoding (should be UTF-8)

2. **"Error loading models"**
   - Verify `aphasia_model.h5` and `vectorizer.pkl` exist in `src/` directory
   - Check file permissions

3. **Low confidence predictions**
   - This is normal for ambiguous cases
   - Consider the text length and quality

### Performance Tips
- Larger text samples generally give more reliable predictions
- Clear, grammatically structured speech improves accuracy
- Remove any non-speech content before uploading

## Dependencies

The application requires:
- streamlit
- tensorflow
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib

Install with:
```bash
pip install -r requirements.txt
```

## Model Performance

Based on comparative analysis:
- **Neural Network**: Accuracy varies (see model training results)
- **Alternative Models**: Logistic Regression, SVM, Random Forest, Naive Bayes available
- **Best Performance**: Check `model_comparison_results.csv` for latest benchmarks

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all required files are present
3. Ensure proper file format for .cha files
4. Review the console output for detailed error messages

# Train all models
cd aphasia-type-classifier/src && python3 model.py

# Run enhanced web app
python3 -m streamlit run app/enhanced_streamlit_app.py

# Test with real data
python3 test_real_data.py