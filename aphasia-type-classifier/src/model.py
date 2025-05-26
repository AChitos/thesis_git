import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model as keras_load_model

# Additional imports for new classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import time

def evaluate_classifier(model, X_test, y_test, model_name, class_names):
    """
    Evaluate a trained classifier and print metrics.
    
    Parameters:
    - model: Trained classifier
    - X_test: Test features
    - y_test: Test labels
    - model_name: Name of the model for display
    - class_names: List of class names
    
    Returns:
    - Dictionary of metrics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_name.upper()}")
    print(f"{'='*60}")
    
    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Prediction Time: {prediction_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'prediction_time': prediction_time
    }

def train_logistic_regression(X_train, y_train, X_test, y_test, class_names):
    """Train and evaluate Logistic Regression classifier."""
    print("\nTraining Logistic Regression...")
    
    start_time = time.time()
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        solver='liblinear'
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Training Time: {training_time:.4f} seconds")
    
    metrics = evaluate_classifier(model, X_test, y_test, "Logistic Regression", class_names)
    metrics['training_time'] = training_time
    
    return model, metrics

def train_svm(X_train, y_train, X_test, y_test, class_names):
    """Train and evaluate SVM classifier."""
    print("\nTraining Support Vector Machine...")
    
    start_time = time.time()
    model = SVC(
        kernel='rbf',
        random_state=42,
        class_weight='balanced',
        gamma='scale'
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Training Time: {training_time:.4f} seconds")
    
    metrics = evaluate_classifier(model, X_test, y_test, "Support Vector Machine", class_names)
    metrics['training_time'] = training_time
    
    return model, metrics

def train_random_forest(X_train, y_train, X_test, y_test, class_names):
    """Train and evaluate Random Forest classifier."""
    print("\nTraining Random Forest...")
    
    start_time = time.time()
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Training Time: {training_time:.4f} seconds")
    
    metrics = evaluate_classifier(model, X_test, y_test, "Random Forest", class_names)
    metrics['training_time'] = training_time
    
    return model, metrics

def train_naive_bayes(X_train, y_train, X_test, y_test, class_names):
    """Train and evaluate Multinomial Naive Bayes classifier."""
    print("\nTraining Multinomial Naive Bayes...")
    
    start_time = time.time()
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Training Time: {training_time:.4f} seconds")
    
    metrics = evaluate_classifier(model, X_test, y_test, "Multinomial Naive Bayes", class_names)
    metrics['training_time'] = training_time
    
    return model, metrics

def train_neural_network(features, labels, class_names):
    """
    Train a neural network for aphasia type classification.

    Parameters:
    - features: Sparse matrix of feature vectors.
    - labels: Array of labels.
    - class_names: List of class names.

    Returns:
    - model: Trained neural network model.
    - metrics: Dictionary of evaluation metrics.
    """
    print("\nTraining Neural Network...")
    
    start_time = time.time()
    
    # Convert sparse matrix to dense
    features = features.toarray()

    # Encode labels as integers
    label_map = {name: idx for idx, name in enumerate(class_names)}
    labels = labels.map(label_map).values
    labels = to_categorical(labels, num_classes=len(class_names))

    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(features, labels.argmax(axis=1))
    y_resampled = to_categorical(y_resampled, num_classes=len(class_names))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(
        'balanced',
        classes=np.array(list(label_map.values())),  # Convert to numpy array
        y=y_resampled.argmax(axis=1)
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights)}

    # Build the neural network
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,  # Stop training after 5 epochs with no improvement
        restore_best_weights=True
    )

    # Learning rate scheduler callback
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduce learning rate by half
        patience=3,  # Wait for 3 epochs with no improvement
        min_lr=1e-6  # Minimum learning rate
    )

    # Train the model with the scheduler
    print("Training the model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,  
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stopping, lr_scheduler],  
        verbose=1
    )
    
    training_time = time.time() - start_time

    # Evaluate the model
    print(f"\nTraining Time: {training_time:.4f} seconds")
    print(f"\n{'='*60}")
    print(f"EVALUATING NEURAL NETWORK")
    print(f"{'='*60}")
    
    start_pred_time = time.time()
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    prediction_time = time.time() - start_pred_time
    
    print(f"Prediction Time: {prediction_time:.4f} seconds")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Predict on the test set
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    
    # Calculate additional metrics
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix - Neural Network')
    plt.show()
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'training_time': training_time,
        'prediction_time': prediction_time
    }

    return model, history, metrics

def load_model(filepath):
    """
    Load a trained Keras model from the specified .h5 file.

    Args:
        filepath (str): Path to the saved model file.

    Returns:
        model: The loaded Keras model.
    """
    return keras_load_model(filepath)

if __name__ == "__main__":
    print("="*80)
    print("APHASIA TYPE CLASSIFICATION - COMPARATIVE ANALYSIS")
    print("="*80)
    print("Loading preprocessed data...")

    # Paths
    processed_data_folder = "/Users/andreaschitos1/Desktop/thesis_git/aphasia-type-classifier/data/processed/"
    model_save_path = "aphasia_model.h5"

    # Load preprocessed data
    data = []
    labels = []
    for file in os.listdir(processed_data_folder):
        if file.endswith(".cha"):
            file_path = os.path.join(processed_data_folder, file)
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                label = lines[0].strip()  # Extract the first line as the label
                text = " ".join(lines[1:])  # Combine the remaining lines as text
                data.append(text)
                labels.append(label)
                print(f"File: {file}, Label: {label}") 

    # Convert to DataFrame
    df = pd.DataFrame({"text": data, "label": labels})
    print(f"\nDataset loaded: {len(df)} samples")
    print(f"Class distribution:")
    print(df['label'].value_counts())

    # Feature extraction using TfidfVectorizer
    print("\nExtracting features using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)  # Limit is set to 5000 features
    features = vectorizer.fit_transform(df["text"])
    labels = df["label"]

    # Save the vectorizer
    joblib.dump(vectorizer, '/Users/andreaschitos1/Desktop/thesis_git/aphasia-type-classifier/src/vectorizer.pkl')
    print(f"Feature matrix shape: {features.shape}")

    # Prepare data for sklearn models (convert sparse matrix to dense array)
    X = features.toarray()
    
    # Encode labels for sklearn models
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    class_names = label_encoder.classes_

    # Apply SMOTE to balance the dataset for sklearn models
    print("\nApplying SMOTE for data balancing...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"Original dataset size: {X.shape[0]}")
    print(f"Resampled dataset size: {X_resampled.shape[0]}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Dictionary to store all model results
    all_results = {}

    # Train and evaluate all models
    print("\n" + "="*80)
    print("TRAINING AND EVALUATING ALL MODELS")
    print("="*80)

    # 1. Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test, class_names)
    all_results['Logistic Regression'] = lr_metrics
    joblib.dump(lr_model, 'logistic_regression_model.pkl')

    # 2. Support Vector Machine
    svm_model, svm_metrics = train_svm(X_train, y_train, X_test, y_test, class_names)
    all_results['SVM'] = svm_metrics
    joblib.dump(svm_model, 'svm_model.pkl')

    # 3. Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test, class_names)
    all_results['Random Forest'] = rf_metrics
    joblib.dump(rf_model, 'random_forest_model.pkl')

    # 4. Naive Bayes
    nb_model, nb_metrics = train_naive_bayes(X_train, y_train, X_test, y_test, class_names)
    all_results['Naive Bayes'] = nb_metrics
    joblib.dump(nb_model, 'naive_bayes_model.pkl')

    # 5. Neural Network
    nn_model, history, nn_metrics = train_neural_network(features, labels, class_names)
    all_results['Neural Network'] = nn_metrics
    nn_model.save(model_save_path)
    print(f"Neural Network model saved to {model_save_path}")

    # Display comparative results
    print("\n" + "="*80)
    print("COMPARATIVE RESULTS SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(all_results).T
    results_df = results_df.round(4)
    
    print("\nPerformance Metrics Comparison:")
    print(results_df[['accuracy', 'precision', 'recall', 'f1_score']])
    
    print("\nTiming Comparison:")
    print(results_df[['training_time', 'prediction_time']])
    
    # Find best performing model
    best_model = results_df['accuracy'].idxmax()
    best_accuracy = results_df['accuracy'].max()
    
    print(f"\nBest performing model: {best_model}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    
    # Save results to CSV
    results_df.to_csv('model_comparison_results.csv')
    print("\nResults saved to 'model_comparison_results.csv'")

    # Plot training history for Neural Network
    if 'history' in locals():
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Neural Network - Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Neural Network - Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Comparison bar chart
        plt.subplot(1, 3, 3)
        models = list(all_results.keys())
        accuracies = [all_results[model]['accuracy'] for model in models]
        plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red', 'purple'])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()
        
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)