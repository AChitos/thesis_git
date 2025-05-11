# This file contains information about the dataset, including the format of the .cha files and instructions for adding new data.

## Dataset Overview

The dataset used in this project consists of transcripts from Greek-speaking patients with aphasia. The transcripts are stored in `.cha` files, which follow a specific format that includes metadata, participant information, and the dialogue itself.

## .cha File Format

Each `.cha` file contains the following sections:

- **Metadata**: This includes information such as the participant IDs, languages, and media associated with the transcript.
- **Dialogue**: The dialogue is structured with lines starting with specific prefixes:
  - `*INV:` indicates the investigator's speech.
  - `*PAR:` indicates the participant's speech.
  - `%wor:` provides a word-level transcription with timing information.
  
The lines that start with `@` are metadata and should be ignored during preprocessing.

## Instructions for Adding New Data

1. **Format the Data**: Ensure that the new transcripts are saved in the `.cha` format, adhering to the structure outlined above.
2. **Place the Files**: Add the new `.cha` files to the `data` directory.
3. **Update the Dataset**: If necessary, update any relevant scripts in the `src` directory to accommodate the new data, particularly in the preprocessing and feature extraction steps.
4. **Retrain the Model**: After adding new data, it may be necessary to retrain the model to include the new examples. Follow the instructions in the main `README.md` for retraining the model.

By following these guidelines, you can contribute new data to the aphasia type classifier project effectively.