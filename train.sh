#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

# python download_multilingual_dataset.py

# python short_files_handler.py

fap loudness-norm data-raw data --clean

cd data-raw
find . -name '*.lab' -exec cp --parents {} ../data/ \;

cd ..

# First Python command
python tools/vqgan/extract_vq.py data \
    --num-workers 1 --batch-size 16 \
    --config-name "firefly_gan_vq" \
    --checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"

# Second Python command
python tools/llama/build_dataset.py \
    --input "data" \
    --output "data/protos" \
    --text-extension .lab \
    --num-workers 16

# Third Python command
python fish_speech/train.py --config-name text2semantic_finetune \
    project=SEP-TOKEN \
    +lora@model.model.lora_config=r_8_alpha_16

python tools/llama/merge_lora.py \
    --lora-config r_8_alpha_16 \
    --base-weight checkpoints/fish-speech-1.5 \
    --lora-weight results/ASS/checkpoints/step_000150000.ckpt \
    --output checkpoints/fish-speech-1.5-multi-lora/

# Optional: Indicate completion
echo "Script finished successfully."
