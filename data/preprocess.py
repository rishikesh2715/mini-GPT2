import os
import re
import requests
import numpy as np
import time # Added for timing

# Make sure tokenizer.py (your custom one) is in the same directory
# or accessible via PYTHONPATH
try:
    from tokenizer import GPT2Tokenizer
except ImportError:
    print("Error: Could not import GPT2Tokenizer. Make sure tokenizer.py is accessible.")
    exit()

# --- Configuration ---
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = "shakespeare" # Relative path where data will be stored
RAW_FILE = os.path.join(DATA_DIR, "input.txt")
CLEAN_FILE = os.path.join(DATA_DIR, "clean_input.txt")
VOCAB_FILE = os.path.join(DATA_DIR, "vocab.json") # For custom tokenizer
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin") # Output token IDs
VAL_BIN = os.path.join(DATA_DIR, "val.bin")   # Output token IDs
TRAIN_FRACTION = 0.9
VOCAB_SIZE = 5000 # Set desired vocab size (e.g., 5000 or 1000)

# --- Functions ---

def download_dataset():
    """Downloads the Tiny Shakespeare dataset if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True) # Ensure the directory exists
    if not os.path.exists(RAW_FILE):
        print(f"Downloading Tiny Shakespeare to {RAW_FILE}...")
        try:
            response = requests.get(DATA_URL)
            response.raise_for_status() # Check for download errors
            with open(RAW_FILE, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading dataset: {e}")
            exit(1) # Exit if download failed
    else:
        print(f"Dataset '{RAW_FILE}' already downloaded.")

def clean_text(text):
    """Cleans the text: normalizes whitespace, handles specific chars, removes non-ASCII."""
    print("Starting text cleaning...")
    start_time = time.time()
    # Normalize different newline types to \n first
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    # Replace multiple spaces with single space (but NOT newlines)
    text = re.sub(r' +', ' ', text)

    # Split into lines, strip whitespace from each line, join back
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip()] # Keep only non-empty lines
    text = '\n'.join(lines)
    # Remove leading/trailing whitespace/newlines from the whole text
    text = text.strip()

    # Replace non-standard characters
    text = text.replace('’', "'").replace('“', '"').replace('”', '"')
    # Remove non-ASCII characters by replacing them with a space
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Collapse any multiple spaces potentially created by the above step
    text = re.sub(r' +', ' ', text)

    end_time = time.time()
    print(f"Text cleaning finished in {end_time - start_time:.2f} seconds.")
    return text

def prepare_data():
    """Full pipeline: Download, Clean, Train Tokenizer, Encode, Save."""
    # 1. Download
    download_dataset()

    # 2. Read and Clean the dataset
    print(f"Reading raw data from {RAW_FILE}...")
    try:
        with open(RAW_FILE, 'r', encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError:
        print(f"Error: Raw file not found at {RAW_FILE}")
        return

    cleaned_data = clean_text(data)
    print(f"Saving cleaned data to {CLEAN_FILE}...")
    try:
        with open(CLEAN_FILE, 'w', encoding='utf-8') as f:
            f.write(cleaned_data)
        print("Cleaned data saved.")
    except IOError as e:
        print(f"Error saving cleaned data: {e}")
        return

    # 3. Initialize and Train Tokenizer
    print(f"Initializing tokenizer with vocab_size={VOCAB_SIZE}...")
    tokenizer = GPT2Tokenizer(vocab_size=VOCAB_SIZE) # Use the config variable

    print("Training tokenizer (this might take a while)...")
    start_time = time.time()
    try:
        tokenizer.train(cleaned_data) # Train on the cleaned data string
        end_time = time.time()
        print(f"Tokenizer training finished in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error during tokenizer training: {e}")
        # Consider raising the exception or exiting depending on desired behavior
        raise # Re-raise the exception to see the full traceback

    # 4. Save Vocabulary
    print(f"Saving vocabulary to {VOCAB_FILE}...")
    try:
        tokenizer.save_vocabulary(VOCAB_FILE)
        print("Vocabulary saved.")
    except IOError as e:
        print(f"Error saving vocabulary: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during vocabulary saving: {e}")
        return


    # 5. Encode Data
    print("Encoding data using the trained tokenizer...")
    start_time = time.time()
    try:
        tokens = tokenizer.encode(cleaned_data)
        end_time = time.time()
        print(f"Data encoding finished in {end_time - start_time:.2f} seconds. Total tokens: {len(tokens)}")
    except Exception as e:
        print(f"Error during data encoding: {e}")
        raise # Re-raise to see traceback

    # 6. Split tokens into Train and Validation sets
    n = int(len(tokens) * TRAIN_FRACTION)
    train_ids = tokens[:n]
    val_ids = tokens[n:]
    print(f"Splitting data: {len(train_ids)} train tokens, {len(val_ids)} validation tokens.")

    # 7. Save Tokenized Data
    print(f"Saving tokenized data to {TRAIN_BIN} and {VAL_BIN}...")
    try:
        # Ensure dtype is appropriate for vocab size
        dtype = np.uint16 if VOCAB_SIZE <= 65535 else np.uint32
        print(f"Using dtype: {dtype}")
        np.array(train_ids, dtype=dtype).tofile(TRAIN_BIN)
        np.array(val_ids, dtype=dtype).tofile(VAL_BIN)
        print("Tokenized data saved.")
    except IOError as e:
        print(f"Error saving tokenized data: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during tokenized data saving: {e}")
        return

    print("-" * 20)
    print("Data preparation complete.")
    print(f"Final vocabulary size reported by tokenizer: {len(tokenizer.token_to_id)}")
    print("-" * 20)

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting data preparation process...")
    prepare_data()
    print("Data preparation process finished.")
