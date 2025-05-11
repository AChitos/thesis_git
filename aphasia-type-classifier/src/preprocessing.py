from typing import List
import re
import os

def preprocess_text(lines: List[str]) -> List[str]:
    """
    Preprocess the input lines from the .cha files.
    
    Args:
        lines (List[str]): List of lines from the .cha file.
        
    Returns:
        List[str]: List of cleaned and normalized lines containing only participant responses.
    """
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
    
    return cleaned_lines

def load_and_preprocess_file(filepath: str) -> List[str]:
    """
    Load a .cha file and preprocess its content.
    
    Args:
        filepath (str): Path to the .cha file.
        
    Returns:
        List[str]: List of cleaned and normalized lines.
    """
    print(f"Processing file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Extract the label from the first line
    label = lines[0].strip()
    
    # Preprocess the remaining lines
    processed_lines = preprocess_text(lines[1:])
    
    # Add the label as the first line in the processed output
    return [label] + processed_lines

data_folder = "/Users/andreaschitos1/Desktop/bachelor_project/wernicke_broca/aphasia-type-classifier/data/"
processed_folder = os.path.join(data_folder, "processed")

# Create the 'processed' folder if it doesn't exist
os.makedirs(processed_folder, exist_ok=True)

# Save the processed file in the 'processed' folder
for file in os.listdir(data_folder):
    if file.endswith(".cha"):
        file_path = os.path.join(data_folder, file)
        processed_lines = load_and_preprocess_file(file_path)
        
        processed_file_path = os.path.join(processed_folder, f"processed_{file}")
        with open(processed_file_path, "w", encoding="utf-8") as out_file:
            out_file.write("\n".join(processed_lines))
        print(f"Saved processed file: {processed_file_path}")