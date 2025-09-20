#!/usr/bin/env python3

import sys
import os
import json
import tempfile
import sqlite3
from pathlib import Path
sys.path.insert(0, 'src')

def test_imports():
    """Test that all GUI imports work correctly"""
    print("Testing GUI v2 imports...")
    
    try:
        from fwhisper_batch.gui_app import (
            SettingsPreset, ProcessingJob, ResultsDatabase, TranscriptionWorker, 
            SettingsManager, MainWindow, main, HELP_TOOLTIPS
        )
        print("✅ GUI v2 imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_settings_preset_defaults():
    """Test that SettingsPreset has correct default values"""
    print("\nTesting SettingsPreset defaults...")
    
    try:
        from fwhisper_batch.gui_app import SettingsPreset
        
        preset = SettingsPreset(name="test")
        
        if preset.max_speakers == 2:
            print("✅ Default max_speakers is correctly set to 2")
        else:
            print(f"❌ Default max_speakers is {preset.max_speakers}, expected 2")
            return False
            
        if preset.model_size == "large-v3":
            print("✅ Default model_size is correctly set to large-v3")
        else:
            print(f"❌ Default model_size is {preset.model_size}, expected large-v3")
            return False
            
        print("✅ SettingsPreset defaults test passed")
        return True
        
    except Exception as e:
        print(f"❌ SettingsPreset defaults test failed: {e}")
        return False

def test_results_database():
    """Test ResultsDatabase functionality"""
    print("\nTesting ResultsDatabase...")
    
    try:
        from fwhisper_batch.gui_app import ResultsDatabase, ProcessingJob, SettingsPreset
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = ResultsDatabase(db_path)
            
            preset = SettingsPreset(name="test_preset")
            job = ProcessingJob(
                file_path=Path("test.wav"),
                settings_preset=preset,
                status="完了",
                result={"processing_time": 10.5, "diarization": {"segments_with_speakers": 3}}
            )
            
            job_id = db.save_result(job)
            
            if job_id > 0:
                print("✅ Result saved successfully")
            else:
                print("❌ Failed to save result")
                return False
                
            results = db.get_recent_results(10)
            
            if len(results) == 1 and results[0]['status'] == '完了':
                print("✅ Result retrieved successfully")
            else:
                print("❌ Failed to retrieve results")
                return False
                
        print("✅ ResultsDatabase test passed")
        return True
        
    except Exception as e:
        print(f"❌ ResultsDatabase test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_help_tooltips():
    """Test that help tooltips are defined"""
    print("\nTesting help tooltips...")
    
    try:
        from fwhisper_batch.gui_app import HELP_TOOLTIPS
        
        required_keys = ["beam_size", "model_size", "compute_type", "min_silence_ms", "max_speakers"]
        
        for key in required_keys:
            if key not in HELP_TOOLTIPS:
                print(f"❌ Missing help tooltip for {key}")
                return False
            if not HELP_TOOLTIPS[key].strip():
                print(f"❌ Empty help tooltip for {key}")
                return False
                
        print("✅ All required help tooltips are defined")
        return True
        
    except Exception as e:
        print(f"❌ Help tooltips test failed: {e}")
        return False

def test_transcription_integration():
    """Test integration with existing transcription logic"""
    print("\nTesting transcription integration...")
    
    try:
        from fwhisper_batch.transcribe_batch import (
            detect_device, resolve_device, resolve_compute_type,
            load_config, is_diarization_enabled, transcribe_one_with_callback
        )
        
        device = detect_device()
        resolved_device = resolve_device("auto")
        compute_type = resolve_compute_type(resolved_device, "auto")
        
        print("✅ Device detection successful")
        print(f"   Detected device: {device}")
        print(f"   Resolved device: {resolved_device}")
        print(f"   Compute type: {compute_type}")
        
        if callable(transcribe_one_with_callback):
            print("✅ transcribe_one_with_callback function available")
        else:
            print("❌ transcribe_one_with_callback function not found")
            return False
        
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

def test_phase_progress_signals():
    """Test that phase progress signals are properly defined"""
    print("\nTesting phase progress signals...")
    
    try:
        from fwhisper_batch.gui_app import TranscriptionWorker
        
        worker = TranscriptionWorker()
        
        if hasattr(worker, 'phase_progress_updated'):
            print("✅ phase_progress_updated signal exists")
        else:
            print("❌ phase_progress_updated signal not found")
            return False
            
        signal = worker.phase_progress_updated
        if signal:
            print("✅ phase_progress_updated signal is properly defined")
        else:
            print("❌ phase_progress_updated signal is not properly defined")
            return False
        
        print("✅ Phase progress signals test passed")
        return True
        
    except Exception as e:
        print(f"❌ Phase progress signals test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing GUI v2 application features...\n")
    
    results = []
    results.append(test_imports())
    results.append(test_settings_preset_defaults())
    results.append(test_results_database())
    results.append(test_help_tooltips())
    results.append(test_transcription_integration())
    results.append(test_phase_progress_signals())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! GUI v2 application is ready.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
