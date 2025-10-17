#!/usr/bin/env python
"""
.env ファイルの読み込みチェックスクリプト

使い方:
    uv run python check_env.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv

def check_env():
    print("=" * 60)
    print("KoeScribe .env ファイル診断ツール")
    print("=" * 60)
    print()
    
    script_dir = Path(__file__).parent
    print(f"✓ スクリプトの場所: {script_dir}")
    print()
    
    env_path = script_dir / ".env"
    print(f"✓ .env ファイルのパス: {env_path}")
    
    if not env_path.exists():
        print("❌ エラー: .env ファイルが見つかりません！")
        print(f"   場所: {env_path}")
        print("   .env ファイルを作成してください。")
        return False
    
    print(f"✓ .env ファイルが存在します")
    print()
    
    print("--- .env ファイルの内容 ---")
    with open(env_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'HUGGINGFACE_TOKEN' in line and '=' in line and not line.strip().startswith('#'):
                parts = line.split('=', 1)
                if len(parts) == 2 and parts[1].strip():
                    token = parts[1].strip()
                    masked_token = token[:10] + "..." + token[-5:] if len(token) > 15 else token
                    print(f"{i}: {parts[0]}={masked_token}")
                else:
                    print(f"{i}: {line}")
            else:
                print(f"{i}: {line}")
    print()
    
    print("--- python-dotenv で読み込み中 ---")
    load_dotenv(dotenv_path=env_path)
    print(f"✓ load_dotenv(dotenv_path={env_path}) 実行完了")
    print()
    
    print("--- 環境変数の確認 ---")
    token = os.environ.get('HUGGINGFACE_TOKEN')
    
    if token:
        print(f"✓ HUGGINGFACE_TOKEN が読み込まれています")
        print(f"  値の先頭: {token[:15]}...")
        print(f"  値の末尾: ...{token[-10:]}")
        print(f"  全体の長さ: {len(token)} 文字")
        
        if token.startswith('hf_'):
            print(f"✓ トークンの形式が正しいです (hf_ で始まっています)")
        else:
            print(f"⚠ 警告: トークンが 'hf_' で始まっていません")
            print(f"  実際の値: {token[:10]}...")
    else:
        print("❌ エラー: HUGGINGFACE_TOKEN が読み込まれていません！")
        print("   .env ファイルの内容を確認してください。")
        return False
    
    print()
    
    print("--- その他の環境変数 ---")
    device = os.environ.get('FWHISPER_DEVICE')
    compute = os.environ.get('FWHISPER_COMPUTE')
    
    if device:
        print(f"  FWHISPER_DEVICE: {device}")
    else:
        print(f"  FWHISPER_DEVICE: (未設定 - auto が使用されます)")
    
    if compute:
        print(f"  FWHISPER_COMPUTE: {compute}")
    else:
        print(f"  FWHISPER_COMPUTE: (未設定 - auto が使用されます)")
    
    print()
    
    print("--- アプリケーションでの読み込みテスト ---")
    print("GUI (gui_app.py) での読み込みをシミュレート:")
    
    os.environ.clear()  # 一度クリア
    project_root = Path(__file__).parent
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path)
    
    token_after = os.environ.get('HUGGINGFACE_TOKEN')
    if token_after:
        print(f"✓ GUI方式で読み込み成功: {token_after[:15]}...")
    else:
        print(f"❌ GUI方式で読み込み失敗")
        return False
    
    print()
    print("=" * 60)
    print("診断完了！")
    print("=" * 60)
    print()
    print("✓ .env ファイルは正しく設定されています。")
    print()
    print("次のステップ:")
    print("1. GUI を起動: uv run koescribe-gui")
    print("2. 話者分離を有効にして処理を実行")
    print("3. エラーが出る場合は、このスクリプトの出力を開発者に共有してください")
    
    return True

if __name__ == "__main__":
    try:
        success = check_env()
        exit(0 if success else 1)
    except Exception as e:
        print()
        print("❌ エラーが発生しました:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
