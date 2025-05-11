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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model as keras_load_model

print("Loading preprocessed data...")

def train_neural_network(features, labels, class_names):
    """
    Train a neural network for aphasia type classification.

    Parameters:
    - features: Sparse matrix of feature vectors.
    - labels: Array of labels.
    - class_names: List of class names.

    Returns:
    - model: Trained neural network model.
    """
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

    # Evaluate the model
    print("\nEvaluating the model...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Predict on the test set
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.show()

    # Print precision, recall, and F1 score
    from evaluate import evaluate_model
    evaluate_model(y_true, y_pred)

    return model, history

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

    # Paths
    processed_data_folder = "/Users/andreaschitos1/Desktop/bachelor_project/wernicke_broca/aphasia-type-classifier/data/processed/"
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

    # Feature extraction using TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=5000)  # Limit is set to 5000 features
    features = vectorizer.fit_transform(df["text"])
    labels = df["label"]

    # Save the vectorizer
    joblib.dump(vectorizer, '/Users/andreaschitos1/Desktop/bachelor_project/wernicke_broca/aphasia-type-classifier/src/vectorizer.pkl')

    # Train the neural network
    class_names = labels.unique()
    model, history = train_neural_network(features, labels, class_names)

    # Save the model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot training history
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()