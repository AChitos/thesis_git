# Aphasia Type Classifier

This project is designed to classify different types of aphasia in Greek-speaking patients using machine learning techniques. The classifier is built using transcripts from patient interviews, which are processed and analyzed to extract relevant linguistic features.

## Project Structure

The project is organized as follows:

- **data/**: Contains the dataset 
- **src/**: Contains the source code for data processing, feature extraction, model training, and evaluation.
  - **preprocessing.py**: Functions for data preprocessing.
  - **feature_extraction.py**: Functions for extracting linguistic features.
  - **model.py**: Functions for model training and inference.
  - **evaluate.py**: Functions for evaluating model performance.
  - **utils.py**: Utility functions for loading, saving data, and logging.

- **app/**: Contains the Streamlit web application code.
  - **streamlit_app.py**: Web app for uploading data, predicting aphasia type, and displaying results.

- **requirements.txt**: Lists the dependencies required for the project.

- **setup.py**: Used for packaging the project with metadata and dependencies.