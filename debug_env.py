#!/usr/bin/env python3
"""
Debug script to check .env file loading and HUGGINGFACE_TOKEN
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, 'src')

def check_env_file():
    """Check if .env file exists and its contents"""
    env_path = Path(".env")
    
    print("🔍 Checking .env file...")
    print(f"Current working directory: {os.getcwd()}")
    print(f".env file path: {env_path.absolute()}")
    print(f".env file exists: {env_path.exists()}")
    
    if env_path.exists():
        print(f".env file size: {env_path.stat().st_size} bytes")
        
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                print(f"Number of lines: {len(lines)}")
                
                for i, line in enumerate(lines, 1):
                    if line.strip():
                        if 'HUGGINGFACE_TOKEN' in line:
                            if '=' in line:
                                key, value = line.split('=', 1)
                                masked_value = value[:8] + '*' * (len(value) - 8) if len(value) > 8 else '*' * len(value)
                                print(f"Line {i}: {key}={masked_value}")
                            else:
                                print(f"Line {i}: {line} (no = found)")
                        else:
                            print(f"Line {i}: {line}")
                    else:
                        print(f"Line {i}: (empty)")
                        
        except Exception as e:
            print(f"❌ Error reading .env file: {e}")
    else:
        print("❌ .env file not found!")
        
        alternatives = [".env.txt", ".env.example", "env", ".environment"]
        for alt in alternatives:
            if Path(alt).exists():
                print(f"Found alternative: {alt}")

def test_dotenv_loading():
    """Test python-dotenv loading"""
    print("\n🧪 Testing python-dotenv loading...")
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv imported successfully")
        
        result = load_dotenv()
        print(f"load_dotenv() result: {result}")
        
        token = os.getenv('HUGGINGFACE_TOKEN')
        if token:
            masked_token = token[:8] + '*' * (len(token) - 8) if len(token) > 8 else '*' * len(token)
            print(f"✅ HUGGINGFACE_TOKEN found: {masked_token}")
        else:
            print("❌ HUGGINGFACE_TOKEN not found in environment")
            
        hugging_vars = {k: v for k, v in os.environ.items() if 'HUGGING' in k.upper()}
        if hugging_vars:
            print("Found HUGGING* environment variables:")
            for k, v in hugging_vars.items():
                masked_v = v[:8] + '*' * (len(v) - 8) if len(v) > 8 else '*' * len(v)
                print(f"  {k}: {masked_v}")
        else:
            print("No HUGGING* environment variables found")
            
    except ImportError:
        print("❌ python-dotenv not installed")
    except Exception as e:
        print(f"❌ Error testing dotenv: {e}")

def test_transcribe_batch_loading():
    """Test how transcribe_batch.py loads the environment"""
    print("\n🔬 Testing transcribe_batch.py environment loading...")
    
    try:
        from fwhisper_batch.transcribe_batch import load_config, is_diarization_enabled
        
        config_path = Path("config.json")
        if config_path.exists():
            config = load_config(config_path)
            print(f"✅ Config loaded: {type(config)}")
            
            diarization_enabled = is_diarization_enabled(config)
            print(f"Diarization enabled: {diarization_enabled}")
            
            token = os.getenv('HUGGINGFACE_TOKEN')
            print(f"HUGGINGFACE_TOKEN in environment: {'Yes' if token else 'No'}")
            
        else:
            print("❌ config.json not found")
            
    except Exception as e:
        print(f"❌ Error testing transcribe_batch: {e}")
        import traceback
        traceback.print_exc()

def suggest_fixes():
    """Suggest potential fixes"""
    print("\n💡 Suggested fixes:")
    print("1. Ensure .env file is in the project root directory")
    print("2. Check .env file format: HUGGINGFACE_TOKEN=hf_your_token_here")
    print("3. No spaces around the = sign")
    print("4. No quotes around the token value")
    print("5. Ensure the file is saved as '.env' (not .env.txt)")
    print("6. Try running from the project root directory")
    print("7. Check file encoding (should be UTF-8)")

if __name__ == "__main__":
    print("🔧 FWhisper Environment Debug Tool")
    print("=" * 40)
    
    check_env_file()
    test_dotenv_loading()
    test_transcribe_batch_loading()
    suggest_fixes()
    
    print("\n" + "=" * 40)
    print("Debug complete!")
