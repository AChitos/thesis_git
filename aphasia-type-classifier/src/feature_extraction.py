from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import joblib

# Load the vectorizer used during training
vectorizer = joblib.load('/Users/andreaschitos1/Desktop/bachelor_project/wernicke_broca/aphasia-type-classifier/src/vectorizer.pkl')

def extract_features(transcripts):
    """
    Extract linguistic features from the given transcripts.

    Parameters:
    transcripts (list of str): List of transcript strings.

    Returns:
    pd.DataFrame: DataFrame containing extracted features.
    """
    features = {
        'sentence_length': [],
        'word_frequency': [],
        'pos_tags': [],
        'lexical_diversity': []
    }

    for transcript in transcripts:
        # Tokenize the transcript into words
        words = word_tokenize(transcript)
        
        # Calculate sentence length
        features['sentence_length'].append(len(words))
        
        # Calculate word frequency
        
        word_count = len(words)
        word_freq = vectorizer.transform([transcript]).toarray().sum(axis=0)
        features['word_frequency'].append(word_freq)
        
        # Part-of-speech tagging
        pos = pos_tag(words)
        features['pos_tags'].append(pos)
        
        # Calculate lexical diversity
        unique_words = len(set(words))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        features['lexical_diversity'].append(lexical_diversity)

    return pd.DataFrame(features)