#!/bin/bash
# KoeScribe GUI をexe形式にビルドするスクリプト（Linux/macOS用）
# 実行: bash build_exe.sh

echo "========================================"
echo "KoeScribe GUI - EXE ビルドスクリプト"
echo "========================================"
echo ""

# 依存関係のインストール確認
echo "[1/4] 依存関係の確認..."
if ! command -v uv &> /dev/null; then
    echo "✗ uv が見つかりません。インストールしてください。"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "✓ uv が見つかりました"

# PyInstallerのインストール
echo "[2/4] PyInstallerのインストール..."
uv sync
if [ $? -ne 0 ]; then
    echo "✗ 依存関係のインストールに失敗しました"
    exit 1
fi
echo "✓ 依存関係のインストール完了"

# 既存のビルドディレクトリをクリーンアップ
echo "[3/4] 既存のビルドファイルをクリーンアップ..."
if [ -d "build" ]; then
    rm -rf build
    echo "✓ build ディレクトリを削除しました"
fi
if [ -d "dist" ]; then
    rm -rf dist
    echo "✓ dist ディレクトリを削除しました"
fi

# PyInstallerでビルド
echo "[4/4] PyInstallerでビルド中..."
echo "  これには数分かかる場合があります..."

uv run pyinstaller koescribe-gui.spec --clean --noconfirm

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ ビルドが完了しました！"
    echo "========================================"
    echo ""
    echo "出力ファイル: dist/KoeScribe-GUI (Linux/macOS)"
    echo ""
    echo "注意事項:"
    echo "  - 初回実行時、モデルファイルがダウンロードされます"
    echo "  - 話者分離機能を使用する場合、.envファイルが必要です"
    echo "  - .envファイルは実行ファイルと同じディレクトリに配置してください"
    echo ""
else
    echo ""
    echo "========================================"
    echo "✗ ビルドに失敗しました"
    echo "========================================"
    echo ""
    echo "エラーログを確認してください"
    exit 1
fi

