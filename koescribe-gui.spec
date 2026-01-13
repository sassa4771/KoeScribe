# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

block_cipher = None

# プロジェクトのルートディレクトリ
# PyInstallerのSPECPATH変数を使用（ビルド時に自動設定される）
try:
    project_root = Path(SPECPATH)
except NameError:
    # SPECPATHが定義されていない場合（直接実行時など）
    project_root = Path(__file__).parent

# faster-whisperのアセットファイルを取得
import faster_whisper
from pathlib import Path as P
fw_path = P(faster_whisper.__file__).parent
assets_path = fw_path / 'assets'

# データファイル（アイコン、faster-whisperのアセットなど）
datas = [
    (str(project_root / 'app_icon.png'), '.'),
]

# faster-whisperのアセットファイルを追加
if assets_path.exists():
    for asset_file in assets_path.glob('*.onnx'):
        datas.append((str(asset_file), 'faster_whisper/assets'))

# 隠しインポート（PyInstallerが自動検出できないモジュール）
hiddenimports = [
    'fwhisper_batch',
    'fwhisper_batch.gui_app',
    'fwhisper_batch.transcribe_batch',
    'fwhisper_batch.video_converter',
    'tools.diarize',
    'tools.merge_speakers',
    'tools.pipeline_all',
    'faster_whisper',
    'pyannote.audio',
    'pyannote.core',
    'ctranslate2',
    'soundfile',
    'numpy',
    'pandas',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'dotenv',
    'tqdm',
    'sqlite3',
    'queue',
    'threading',
]

a = Analysis(
    ['src/fwhisper_batch/gui_app.py'],
    pathex=[str(project_root / 'src'), str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
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
    name='KoeScribe-GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUIアプリなのでコンソールを非表示
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / 'app_icon.png') if (project_root / 'app_icon.png').exists() else None,
)

