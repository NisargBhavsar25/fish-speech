from datasets import load_dataset
import dotenv
import os
import soundfile as sf
from tqdm.auto import tqdm
import time
from collections import defaultdict
import re
import shutil
import gc

# Load environment variables
dotenv.load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Verify token exists
if not HF_TOKEN:
    print("❌ Error: HUGGINGFACE_TOKEN not found in environment variables")
    print("Please create a .env file with your Hugging Face token:")
    print("HUGGINGFACE_TOKEN=your_token_here")
    exit(1)

print(f"✅ Token loaded: {HF_TOKEN[:10]}...")

# Set download directory
download_dir = "./dataset_cache"
os.makedirs(download_dir, exist_ok=True)

# Define the languages we want to process
languages = ["Hindi", "Marathi", "Tamil", "Telugu", "Malayalam", "Kannada"]

lang_code_map = {
    "Hindi": "hi",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn"
}

target_hours_per_language = 200
target_seconds_per_language = target_hours_per_language * 3600

# Define the base directory for saving datasets
base_dir = "./data-raw"

def clear_cache():
    """Clear the dataset cache to save space"""
    try:
        if os.path.exists(download_dir):
            # Calculate cache size before deletion
            cache_size = get_directory_size(download_dir)
            print(f"🗑️ Clearing cache ({cache_size:.2f} MB)...")
            
            # Delete the entire cache directory
            shutil.rmtree(download_dir)
            
            # Recreate the cache directory for next use
            os.makedirs(download_dir, exist_ok=True)
            
            print(f"✅ Cache cleared successfully")
        else:
            print("ℹ️ No cache directory found to clear")
            
        # Force garbage collection to free memory
        gc.collect()
        
    except Exception as e:
        print(f"❌ Error clearing cache: {str(e)}")

def get_directory_size(directory):
    """Calculate the total size of a directory in MB"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # Convert to MB
    except Exception as e:
        print(f"❌ Error calculating directory size: {e}")
        return 0

def sanitize_speaker_id(speaker_id):
    """Clean speaker ID to make it suitable for folder names"""
    speaker_id = str(speaker_id)
    speaker_id = re.sub(r'[<>:"/\\|?*\s]', '_', speaker_id)
    speaker_id = re.sub(r'_+', '_', speaker_id)
    speaker_id = speaker_id.strip('_')
    if not speaker_id:
        speaker_id = "unknown_speaker"
    return speaker_id

def create_directories():
    try:
        abs_base_dir = os.path.abspath(base_dir)
        os.makedirs(abs_base_dir, exist_ok=True)
        
        print(f"✅ Created directory: {abs_base_dir}")
        
        # Test write permissions
        test_file = os.path.join(abs_base_dir, "test_write.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"✅ Write permissions confirmed")
        
        return abs_base_dir
    except Exception as e:
        print(f"❌ Error creating directories: {str(e)}")
        return None

# def test_dataset_access():
#     """Test if we can access the IndicVoices dataset"""
#     try:
#         print("🔍 Testing dataset access...")
#         test_ds = load_dataset(
#             "ai4bharat/indicvoices_r",
#             "hindi",
#             split="train[:1]",
#             token=HF_TOKEN,
#             cache_dir=download_dir
#         )
#         print(f"✅ Dataset access successful! Sample keys: {list(test_ds[0].keys())}")
#         return True
#     except Exception as e:
#         print(f"❌ Dataset access failed: {str(e)}")
#         return False

def save_speaker_based_dataset(languages, abs_base_dir, target_seconds_per_language):
    # Test dataset access first
    # if not test_dataset_access():
    #     print("❌ Cannot access dataset. Please check your token and internet connection.")
    #     return
    
    # Clear initial test cache
    clear_cache()
    
    # Track unique speakers
    all_speakers = set()
    
    # Stats tracking
    total_processed = 0
    total_saved_files = 0
    start_time = time.time()
    
    # Process each language
    for lang_idx, lang in enumerate(languages):
        print(f"\n{'='*60}")
        print(f"Processing language {lang_idx+1}/{len(languages)}: {lang.upper()}")
        print(f"{'='*60}")
        
        # Check available disk space before processing
        if hasattr(shutil, 'disk_usage'):
            _, _, free_space = shutil.disk_usage(abs_base_dir)
            free_gb = free_space / (1024**3)
            print(f"💽 Available disk space: {free_gb:.2f} GB")
        
        try:
            # Load the dataset for this language
            print(f"📥 Loading dataset for {lang}...")
            ds = load_dataset(
                "ai4bharat/indicvoices_r",
                lang,
                token=HF_TOKEN,
                cache_dir=download_dir
            )
            print(f"✅ Dataset loaded for {lang}. Available splits: {list(ds.keys())}")
            
            lang_processed = 0
            lang_duration = 0
            lang_saved_files = 0
            
            # Process train split first (usually largest)
            splits_to_process = ["train"] + [s for s in ds.keys() if s != "train"]
            
            for split in splits_to_process:
                if lang_duration >= target_seconds_per_language:
                    print(f"🎯 Target duration reached for {lang}")
                    break
                
                if split not in ds:
                    continue
                    
                split_data = ds[split]
                print(f"\n📊 Processing {lang}-{split}: {len(split_data)} samples")
                
                split_saved = 0
                split_duration = 0
                
                # Track file counters per speaker for this split
                speaker_file_counters = defaultdict(int)
                
                # Process samples with progress bar
                with tqdm(total=len(split_data), desc=f"{lang}-{split}", unit="samples") as pbar:
                    
                    for idx in range(len(split_data)):
                        if lang_duration >= target_seconds_per_language:
                            break
                            
                        try:
                            example = split_data[idx]
                            
                            # Extract audio data
                            audio_array = None
                            sampling_rate = None
                            
                            # Try different possible audio field names
                            audio_fields = ['audio', 'audio_filepath', 'sound', 'wav']
                            for field in audio_fields:
                                if field in example:
                                    audio_data = example[field]
                                    if isinstance(audio_data, dict) and 'array' in audio_data:
                                        audio_array = audio_data['array']
                                        sampling_rate = audio_data['sampling_rate']
                                        break
                                    else:
                                        try:
                                            audio_array, sampling_rate = sf.read(audio_data)
                                            break
                                        except:
                                            continue
                            
                            if audio_array is None:
                                pbar.update(1)
                                continue
                            
                            # Calculate duration
                            audio_duration = len(audio_array) / sampling_rate
                            
                            # Skip very short audio
                            if audio_duration < 0.5:
                                pbar.update(1)
                                continue
                            
                            # Check if we would exceed target
                            if lang_duration + audio_duration > target_seconds_per_language:
                                remaining = target_seconds_per_language - lang_duration
                                if remaining < 10:  # Less than 10 seconds remaining
                                    break
                            
                            # Get and sanitize speaker info
                            original_speaker = example.get("speaker_id", f"{lang}_unknown_{idx}")
                            speaker_folder_name = f"{lang}_{sanitize_speaker_id(original_speaker)}"
                            
                            # Track all speakers
                            all_speakers.add(speaker_folder_name)
                            
                            # Create speaker directory
                            speaker_dir = os.path.join(abs_base_dir, speaker_folder_name)
                            os.makedirs(speaker_dir, exist_ok=True)
                            
                            # Generate file counter for this speaker
                            speaker_file_counters[speaker_folder_name] += 1
                            file_counter = speaker_file_counters[speaker_folder_name]
                            
                            # Create timestamp-like filename (using duration and counter)
                            start_time_sim = file_counter * 10  # Simulate timestamps
                            end_time_sim = start_time_sim + audio_duration
                            file_base = f"{start_time_sim:.2f}-{end_time_sim:.2f}"
                            
                            # Save audio file
                            audio_file = os.path.join(speaker_dir, f"{file_base}.wav")
                            lab_file = os.path.join(speaker_dir, f"{file_base}.lab")
                            
                            try:
                                # Save audio file
                                sf.write(audio_file, audio_array, sampling_rate)
                                
                                # Save transcription to .lab file
                                text = example.get("text", "").strip()
                                lang_code = lang_code_map.get(lang, "unknown")
                                text = f"<{lang_code}>" + text + f"</{lang_code}>"
                                with open(lab_file, "w", encoding="utf-8") as f:
                                    f.write(text)
                                
                                # Verify files were saved
                                if (os.path.exists(audio_file) and os.path.getsize(audio_file) > 0 and
                                    os.path.exists(lab_file) and os.path.getsize(lab_file) > 0):
                                    
                                    # Update counters
                                    lang_processed += 1
                                    lang_duration += audio_duration
                                    lang_saved_files += 1
                                    split_saved += 1
                                    split_duration += audio_duration
                                    
                                    # Update progress bar with current stats
                                    pbar.set_postfix(
                                        saved=split_saved,
                                        duration=f"{split_duration/3600:.2f}h",
                                        total_duration=f"{lang_duration/3600:.2f}h",
                                        speakers=len(all_speakers)
                                    )
                                
                            except Exception as save_e:
                                print(f"❌ Save error: {save_e}")
                        
                        except Exception as process_e:
                            print(f"❌ Process error: {process_e}")
                        
                        pbar.update(1)
                
                print(f"✅ {lang}-{split}: {split_saved} files saved, {split_duration/3600:.2f}h")
            
            total_processed += lang_processed
            total_saved_files += lang_saved_files
            
            print(f"🎉 {lang} completed: {lang_saved_files} files, {lang_duration/3600:.2f}h")
            
            # Clear dataset reference and force garbage collection
            del ds
            gc.collect()
            
            # Clear cache after completing each language
            print(f"\n🧹 Cleaning up cache for {lang}...")
            clear_cache()
            
        except Exception as e:
            print(f"❌ Failed to process {lang}: {str(e)}")
            # Still clear cache even if processing failed
            clear_cache()
            continue
    
    # Final summary
    elapsed = time.time() - start_time
    print(f"\n🏁 FINAL SUMMARY:")
    print(f"⏱️  Processing time: {int(elapsed//60)}m {int(elapsed%60)}s")
    print(f"📊 Total files saved: {total_saved_files}")
    print(f"👥 Total speakers: {len(all_speakers)}")
    print(f"📁 Speaker directories created:")
    
    # List all speaker directories
    if os.path.exists(abs_base_dir):
        for speaker_dir in sorted(os.listdir(abs_base_dir)):
            if os.path.isdir(os.path.join(abs_base_dir, speaker_dir)):
                file_count = len([f for f in os.listdir(os.path.join(abs_base_dir, speaker_dir)) if f.endswith('.wav')])
                print(f"   {speaker_dir}: {file_count} audio files")
    
    # Final cache cleanup
    print(f"\n🧹 Final cache cleanup...")
    clear_cache()

# Run the process
print("🚀 Starting speaker-based dataset creation...")
abs_base_dir = create_directories()

if abs_base_dir:
    save_speaker_based_dataset(languages, abs_base_dir, target_seconds_per_language)
else:
    print("❌ Cannot create directories. Process aborted.")
