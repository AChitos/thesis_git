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

# Update this path to point to your actual data location
data_folder = "/Users/andreaschitos1/Desktop/thesis_git/aphasia-type-classifier/data/"
processed_folder = os.path.join(data_folder, "processed")

# Create the 'processed' folder if it doesn't exist
os.makedirs(processed_folder, exist_ok=True)

# Counter for processed files
processed_count = 0
skipped_count = 0

# Walk through all directories and subdirectories
for root, dirs, files in os.walk(data_folder):
    for file in files:
        # Case-insensitive check for .cha extension
        if file.lower().endswith(".cha"):
            # Get relative path from data_folder to maintain structure
            rel_path = os.path.relpath(root, data_folder)
            
            # Skip if we're already in the processed folder
            if "processed" in rel_path:
                continue
                
            file_path = os.path.join(root, file)
            
            try:
                processed_lines = load_and_preprocess_file(file_path)
                
                # Create subdirectory structure in processed folder if needed
                if rel_path != ".":
                    target_dir = os.path.join(processed_folder, rel_path)
                    os.makedirs(target_dir, exist_ok=True)
                    processed_file_path = os.path.join(target_dir, f"processed_{file}")
                else:
                    processed_file_path = os.path.join(processed_folder, f"processed_{file}")
                
                with open(processed_file_path, "w", encoding="utf-8") as out_file:
                    out_file.write("\n".join(processed_lines))
                print(f"Saved processed file: {processed_file_path}")
                processed_count += 1
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                skipped_count += 1

print(f"Processing complete. Processed {processed_count} files. Skipped {skipped_count} files.")