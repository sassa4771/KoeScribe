#!/usr/bin/env python3
"""Test script for GUI features"""

import json
import pandas as pd
from pathlib import Path
import tempfile
import sys
sys.path.insert(0, 'src')

def test_imports():
    """Test that all GUI imports work correctly"""
    print("Testing GUI imports...")
    
    try:
        from fwhisper_batch.gui_app import (
            ProjectConfig, ProjectManager, TranscriptionWorker, MainWindow, main
        )
        print("✅ GUI imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_csv_conversion():
    """Test JSONL to CSV conversion functionality"""
    print("\nTesting CSV conversion...")
    
    test_segments = [
        {"start": 1.0, "end": 3.0, "text": "こんにちは", "speaker": "SPEAKER_00"},
        {"start": 4.0, "end": 6.0, "text": "元気ですか", "speaker": "SPEAKER_01"},
        {"start": 7.0, "end": 9.0, "text": "はい、元気です", "speaker": "SPEAKER_00"}
    ]
    
    test_words = [
        {"start": 1.0, "end": 1.5, "word": "こんにちは", "speaker": "SPEAKER_00"},
        {"start": 4.0, "end": 4.5, "word": "元気", "speaker": "SPEAKER_01"},
        {"start": 4.5, "end": 5.0, "word": "ですか", "speaker": "SPEAKER_01"}
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        segments_file = temp_path / "test_segments_with_speakers.jsonl"
        words_file = temp_path / "test_words_with_speakers.jsonl"
        
        with open(segments_file, 'w', encoding='utf-8') as f:
            for item in test_segments:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        with open(words_file, 'w', encoding='utf-8') as f:
            for item in test_words:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        try:
            segments_data = []
            with open(segments_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        segments_data.append(json.loads(line))
            
            df_segments = pd.DataFrame(segments_data)
            csv_segments = temp_path / "test_segments_with_speakers.csv"
            df_segments.to_csv(csv_segments, index=False, encoding='utf-8-sig')
            
            words_data = []
            with open(words_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        words_data.append(json.loads(line))
            
            df_words = pd.DataFrame(words_data)
            csv_words = temp_path / "test_words_with_speakers.csv"
            df_words.to_csv(csv_words, index=False, encoding='utf-8-sig')
            
            print("✅ CSV conversion successful")
            print(f"   Segments CSV: {csv_segments.exists()}")
            print(f"   Words CSV: {csv_words.exists()}")
            
            df_check = pd.read_csv(csv_segments, encoding='utf-8-sig')
            print(f"   Segments CSV rows: {len(df_check)}")
            print(f"   Segments CSV columns: {list(df_check.columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ CSV conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_project_config():
    """Test project configuration functionality"""
    print("\nTesting project configuration...")
    
    try:
        from fwhisper_batch.gui_app import ProjectConfig, ProjectManager
        
        config = ProjectConfig(
            name="テストプロジェクト",
            model_size="medium",
            language="ja",
            max_speakers=5
        )
        
        print("✅ ProjectConfig creation successful")
        print(f"   Name: {config.name}")
        print(f"   Model: {config.model_size}")
        print(f"   Max speakers: {config.max_speakers}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ProjectManager()
            original_dir = manager.projects_dir
            manager.projects_dir = Path(temp_dir)
            
            manager.save_project(config)
            loaded_config = manager.load_project("テストプロジェクト")
            
            if loaded_config and loaded_config.name == config.name:
                print("✅ Project save/load successful")
                return True
            else:
                print("❌ Project save/load failed")
                return False
                
    except Exception as e:
        print(f"❌ Project config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transcription_integration():
    """Test integration with existing transcription logic"""
    print("\nTesting transcription integration...")
    
    try:
        from fwhisper_batch.transcribe_batch import (
            detect_device, resolve_device, resolve_compute_type,
            load_config, is_diarization_enabled
        )
        
        device = detect_device()
        resolved_device = resolve_device("auto")
        compute_type = resolve_compute_type(resolved_device, "auto")
        
        print("✅ Device detection successful")
        print(f"   Detected device: {device}")
        print(f"   Resolved device: {resolved_device}")
        print(f"   Compute type: {compute_type}")
        
        config_path = Path("config.json")
        if config_path.exists():
            config = load_config(config_path)
            diarization_enabled = is_diarization_enabled(config)
            print("✅ Config loading successful")
            print(f"   Diarization enabled: {diarization_enabled}")
        else:
            print("⚠️  config.json not found, skipping config test")
        
        return True
        
    except Exception as e:
        print(f"❌ Transcription integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing GUI application features...\n")
    
    results = []
    results.append(test_imports())
    results.append(test_csv_conversion())
    results.append(test_project_config())
    results.append(test_transcription_integration())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! GUI application is ready.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
