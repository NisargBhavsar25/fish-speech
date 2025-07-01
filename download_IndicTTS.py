import os
from datasets import load_from_disk
import soundfile as sf
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Base directory where the raw Hugging Face datasets are stored
BASE_INPUT_DIR = "/home/ubuntu/Dikshit/IndicTTS_Datasets"

# Directory where the formatted dataset will be saved
OUTPUT_DIR = "data-raw"

# List of languages to process
LANGUAGES = ['hindi', 'gujrati', 'marathi', 'telugu', 'tamil', 'malayalam', 'kannada', 'bengali']
# lang_code_map
lang_code_map = {
    'hindi': 'hi',
    'gujrati': 'gu',
    'marathi': 'mr',
    'telugu': 'te',
    'tamil': 'ta',
    'malayalam': 'ml',
    'kannada': 'kn',
    'bengali': 'bn'
}

# LANGUAGES = ['gujrati']

# --- Main Script ---

def format_dataset():
    """
    Loads IndicTTS datasets, filters them, and reorganizes them into the specified format.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created output directory: {OUTPUT_DIR}")

    total_processed_files = 0
    total_skipped_files = 0

    for lang in LANGUAGES:
        dataset_path = os.path.join(BASE_INPUT_DIR, f"{lang}_complete")
        
        if not os.path.exists(dataset_path):
            logging.warning(f"Dataset path not found for language '{lang}': {dataset_path}. Skipping.")
            continue
            
        logging.info(f"Processing language: {lang}")
        
        try:
            # Load the dataset for the current language
            dataset = load_from_disk(dataset_path)['train']
            logging.info(f"Loaded dataset for {lang} with {len(dataset)} samples.")
        except Exception as e:
            logging.error(f"Failed to load dataset for {lang} from {dataset_path}. Error: {e}")
            continue

        for sample in tqdm(dataset, desc=f"Formatting {lang}"):
            try:
                audio_dict = sample['audio']
                text = sample['text']
                gender = sample['gender']
                
                # Calculate audio duration
                sampling_rate = audio_dict['sampling_rate']
                duration = len(audio_dict['array']) / sampling_rate
                
                # Skip files with duration less than 1 second
                if duration < 1.0:
                    total_skipped_files += 1
                    continue
                
                # Create speaker ID from gender and language
                speaker_id = f"{gender}_{lang}"
                speaker_dir = os.path.join(OUTPUT_DIR, speaker_id)
                os.makedirs(speaker_dir, exist_ok=True)
                
                # Assumption: Derive the 'start-end' filename from the original audio file path.
                # For example, if path is '/path/to/21.15-26.44.wav', base_filename becomes '21.15-26.44'
                original_path = audio_dict.get('path')
                if not original_path:
                    logging.warning("Sample is missing the 'path' key in the audio dictionary. Skipping.")
                    total_skipped_files += 1
                    continue
                
                base_filename = os.path.splitext(os.path.basename(original_path))[0]
                
                # Define output paths for the new .wav and .lab files
                wav_path = os.path.join(speaker_dir, f"{base_filename}.wav")
                lab_path = os.path.join(speaker_dir, f"{base_filename}.lab")
                
                # Write the audio data to a .wav file
                sf.write(wav_path, audio_dict['array'], sampling_rate)
                
                # Write the transcription to a .lab file
                # add language code to the text
                lang_code = lang_code_map.get(lang, lang)  # Default to lang if not found
                text = f"<{lang_code}> {text} </{lang_code}>"
                with open(lab_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                total_processed_files += 1

            except Exception as e:
                logging.error(f"Error processing a sample for {lang}. Error: {e}")
                total_skipped_files += 1
    
    logging.info("--- Processing Summary ---")
    logging.info(f"Dataset formatting complete.")
    logging.info(f"Total files successfully processed: {total_processed_files}")
    logging.info(f"Total files skipped (duration < 1s or errors): {total_skipped_files}")


if __name__ == '__main__':
    format_dataset()
