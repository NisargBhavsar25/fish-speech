import os
import soundfile as sf
import shutil

audio_dir = 'data-raw'
min_length_seconds = 1.0  # Changed to 1 second
short_audio_files = []

print("Scanning for audio files...")
total_files = 0
processed_files = 0

# First pass: count total files
for root, dirs, files in os.walk(audio_dir):
    for file in files:
        if file.endswith('.wav') or file.endswith('.mp3') or file.endswith('.flac'):
            total_files += 1

print(f"Found {total_files} audio files. Analyzing length...")

# Second pass: process files
for root, dirs, files in os.walk(audio_dir):
    for file in files:
        if file.endswith('.wav') or file.endswith('.mp3') or file.endswith('.flac'):
            file_path = os.path.join(root, file)
            processed_files += 1
            
            if processed_files % 100 == 0 or processed_files == total_files:
                print(f"Progress: {processed_files}/{total_files} files processed")
            
            try:
                data, samplerate = sf.read(file_path)
                duration_seconds = len(data) / samplerate  # Calculate duration in seconds
                if duration_seconds < min_length_seconds:
                    short_audio_files.append(file_path)
            except Exception:
                continue

print(f"\nAnalysis complete! Found {len(short_audio_files)} short files (< {min_length_seconds} seconds)")

if short_audio_files:
    print("Short files:", short_audio_files)
    
    print(f"\nDeleting {len(short_audio_files)} short files...")
    
    deleted_audio = 0
    deleted_lab = 0
    
    for i, file_path in enumerate(short_audio_files, 1):
        # Delete the audio file
        filename = os.path.basename(file_path)
        os.remove(file_path)
        deleted_audio += 1
        print(f"[{i}/{len(short_audio_files)}] Deleted audio: {filename}")
        
        # Check for corresponding .lab file and delete it too
        base_name = os.path.splitext(file_path)[0]
        lab_file_path = base_name + '.lab'
        
        if os.path.exists(lab_file_path):
            lab_filename = os.path.basename(lab_file_path)
            os.remove(lab_file_path)
            deleted_lab += 1
            print(f"    └── Deleted lab: {lab_filename}")
    
    print(f"\n✅ Deletion complete!")
    print(f"   Audio files deleted: {deleted_audio}")
    print(f"   Lab files deleted: {deleted_lab}")
else:
    print("No short files found. Nothing to delete.")