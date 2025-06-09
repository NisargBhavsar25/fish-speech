#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

# First Python command
python tools/vqgan/extract_vq.py data-as \
    --num-workers 1 --batch-size 16 \
    --config-name "firefly_gan_vq" \
    --checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"

# Second Python command
python tools/llama/build_dataset.py \
    --input "data-as" \
    --output "data-as/protos" \
    --text-extension .lab \
    --num-workers 16

# Third Python command
python fish_speech/train.py --config-name text2semantic_finetune \
    project=ASS \
    +lora@model.model.lora_config=r_8_alpha_16

# Optional: Indicate completion
echo "Script finished successfully."
