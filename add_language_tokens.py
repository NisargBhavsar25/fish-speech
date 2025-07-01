from fish_speech.tokenizer import FishTokenizer
import os

def create_language_tokenizer():
    # Load the original Fish Speech tokenizer
    original_tokenizer_path = "checkpoints/fish-speech-1.5/tokenizer.tiktoken"
    
    if not os.path.exists(original_tokenizer_path):
        print("❌ Error: Fish Speech 1.5 tokenizer not found!")
        print("Please download it first:")
        print("huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5")
        return None
    
    print("📥 Loading original Fish Speech tokenizer...")
    # Create tokenizer with language tokens (your modified tokenizer.py will be used)
    tokenizer = FishTokenizer(original_tokenizer_path)
    
    # Save the updated tokenizer
    output_dir = "checkpoints/fish-speech-1.5-language-tokens"
    print(f"💾 Saving language tokenizer to: {output_dir}")
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Created language tokenizer successfully!")
    print(f"📊 Total vocabulary size: {len(tokenizer.all_special_tokens_with_ids)}")
    print(f"🎭 Added {len(LANGUAGE_TOKENS)} language tokens")
    
    return tokenizer

if __name__ == "__main__":
    # Import the language tokens from the modified tokenizer
    from fish_speech.tokenizer import LANGUAGE_TOKENS
    
    print("🎭 language Tokens to be added:")
    for token in LANGUAGE_TOKENS:
        print(f"  {token}")
    print(f"\nTotal: {len(LANGUAGE_TOKENS)} language tokens\n")
    
    tokenizer = create_language_tokenizer()
    
    if tokenizer:
        print("🎉 language tokenizer is ready for training!")
        print("\nNext steps:")
        print("1. Use 'checkpoints/fish-speech-1.5-language' as your tokenizer path")
        print("2. Update your training vocabulary size to include language tokens")
        print("3. Start fine-tuning with your language dataset")
