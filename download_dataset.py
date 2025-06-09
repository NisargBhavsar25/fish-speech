from datasets import load_dataset
import dotenv
import os
import soundfile as sf
from collections import defaultdict
from tqdm.auto import tqdm
import time
import subprocess

# Load environment variables
dotenv.load_dotenv() 
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Set base directory for raw data
base_dir = "data-raw-as"

def save_dataset(ds, base_dir):
    speaker_counters = defaultdict(int)
    total_processed = 0
    total_errors = 0
    start_time = time.time()

    # Process each split
    for split in ds.keys():
        print(f"\nProcessing split '{split}' with {len(ds[split])} samples...")
        
        for example in tqdm(ds[split], desc=f"Processing {split}"):
            try:
                # Extract speaker ID and create directory
                speaker = str(example["speaker_id"])
                speaker_dir = os.path.join(base_dir, speaker)
                os.makedirs(speaker_dir, exist_ok=True)

                # Generate unique filename using speaker-specific counter
                speaker_counters[speaker] += 1
                base_name = f"{speaker_counters[speaker]:04d}"
                
                # Handle audio data
                audio_data = example.get('audio', {}) or example.get('audio_filepath', {})
                audio_array = audio_data.get('array')
                sampling_rate = audio_data.get('sampling_rate', 44100)

                if audio_array is None:
                    raise ValueError("No audio data found in example")

                # Save audio file
                audio_path = os.path.join(speaker_dir, f"{base_name}.wav")
                sf.write(audio_path, audio_array, sampling_rate)

                # Create corresponding lab file
                lab_path = os.path.join(speaker_dir, f"{base_name}.lab")
                with open(lab_path, "w", encoding="utf-8") as f:
                    f.write(example["text"].strip())

                total_processed += 1

            except Exception as e:
                total_errors += 1
                continue

    # Print statistics
    elapsed = time.time() - start_time
    print(f"\nProcessed {total_processed} samples with {total_errors} errors")
    print(f"Elapsed time: {int(elapsed//60)}m {int(elapsed%60)}s")

    # Apply loudness normalization
    print("\nApplying loudness normalization...")
    result = subprocess.run(
        ["fap", "loudness-norm", base_dir, "data-as", "--clean"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("Successfully normalized audio files in 'data' directory")
    else:
        print(f"Normalization error: {result.stderr}")

# Load dataset and process
ds = load_dataset(
    "ai4bharat/IndicVoices",
    "assamese",
    token=HF_TOKEN,
    cache_dir="/home/ubuntu/Projects/nisarg/XTTSv2-Finetuning-for-New-Languages/dataset"
)

save_dataset(ds, base_dir)