#!/usr/bin/env python3
"""
Build script for creating standalone executable of FWhisper GUI
"""

import os
import sys
import subprocess
from pathlib import Path

def install_pyinstaller():
    """Install PyInstaller if not already installed"""
    try:
        import PyInstaller
        print("✅ PyInstaller already installed")
    except ImportError:
        print("📦 Installing PyInstaller...")
        subprocess.run(["uv", "add", "pyinstaller"], check=True)

def build_executable():
    """Build the executable using PyInstaller"""
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    path_sep = ";" if os.name == "nt" else ":"
    
    cmd = [
        "pyinstaller",
        "--onefile",                    # Single executable file
        "--windowed",                   # No console window (GUI app)
        "--name=FWhisper-GUI",          # Executable name
        "--paths=src",                  # Add src to Python path
        "--hidden-import=PySide6",
        "--hidden-import=faster_whisper",
        "--hidden-import=pyannote.audio",
        "--hidden-import=soundfile",
        "--hidden-import=webrtcvad",
        "--hidden-import=fwhisper_batch",
        "--hidden-import=fwhisper_batch.transcribe_batch",
        "--hidden-import=fwhisper_batch.gui_app",
        "--collect-all=PySide6",
        "--collect-all=faster_whisper",
        "src/fwhisper_batch/gui_app.py"
    ]
    
    icon_path = project_root / "icon.ico"
    if icon_path.exists():
        cmd.insert(-1, f"--icon={icon_path}")
    
    config_path = project_root / "config.json"
    if config_path.exists():
        cmd.insert(-1, f"--add-data=config.json{path_sep}.")
    
    env_path = project_root / ".env"
    if env_path.exists():
        cmd.insert(-1, f"--add-data=.env{path_sep}.")
    
    print("🔨 Building executable...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Build completed successfully!")
        
        exe_ext = ".exe" if os.name == "nt" else ""
        exe_path = f"dist/FWhisper-GUI{exe_ext}"
        print(f"📁 Executable location: {exe_path}")
        
        print("\n📋 Distribution Instructions:")
        print(f"1. Copy the executable from {exe_path}")
        if config_path.exists() or env_path.exists():
            print("2. Include config.json and .env files in the same directory")
        print("3. Ensure target machine has required audio codecs")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False
    
    return True

def create_spec_file():
    """Create a custom .spec file for more control"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/fwhisper_batch/gui_app.py'],
    pathex=['src'],
    binaries=[],
    datas=[
        ('config.json', '.'),
        ('.env', '.'),
    ],
    hiddenimports=[
        'PySide6',
        'faster_whisper',
        'pyannote.audio',
        'soundfile',
        'webrtcvad',
        'numpy',
        'pandas',
        'sqlite3',
        'fwhisper_batch',
        'fwhisper_batch.transcribe_batch',
        'fwhisper_batch.gui_app',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FWhisper-GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''
    
    with open("fwhisper-gui.spec", "w", encoding="utf-8") as f:
        f.write(spec_content)
    
    print("📝 Created fwhisper-gui.spec file")

if __name__ == "__main__":
    print("🚀 FWhisper GUI Executable Builder")
    print("=" * 40)
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Warning: Not in a virtual environment")
    
    install_pyinstaller()
    
    create_spec_file()
    
    success = build_executable()
    
    if success:
        print("\n🎉 Build process completed!")
        print("Run the executable from dist/FWhisper-GUI.exe")
    else:
        print("\n❌ Build process failed!")
        sys.exit(1)
