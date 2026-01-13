# KoeScribe GUI をexe形式にビルドするスクリプト
# PowerShellで実行: .\build_exe.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "KoeScribe GUI - EXE ビルドスクリプト" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 依存関係のインストール確認
Write-Host "[1/4] 依存関係の確認..." -ForegroundColor Yellow
try {
    uv --version | Out-Null
    Write-Host "✓ uv が見つかりました" -ForegroundColor Green
} catch {
    Write-Host "✗ uv が見つかりません。インストールしてください。" -ForegroundColor Red
    Write-Host "  powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor Yellow
    exit 1
}

# PyInstallerのインストール
Write-Host "[2/4] PyInstallerのインストール..." -ForegroundColor Yellow
uv sync
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ 依存関係のインストールに失敗しました" -ForegroundColor Red
    exit 1
}
Write-Host "✓ 依存関係のインストール完了" -ForegroundColor Green

# 既存のビルドディレクトリをクリーンアップ
Write-Host "[3/4] 既存のビルドファイルをクリーンアップ..." -ForegroundColor Yellow
if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
    Write-Host "✓ build ディレクトリを削除しました" -ForegroundColor Green
}
if (Test-Path "dist") {
    Remove-Item -Recurse -Force "dist"
    Write-Host "✓ dist ディレクトリを削除しました" -ForegroundColor Green
}

# PyInstallerでビルド
Write-Host "[4/4] PyInstallerでビルド中..." -ForegroundColor Yellow
Write-Host "  これには数分かかる場合があります..." -ForegroundColor Gray

uv run pyinstaller koescribe-gui.spec --clean --noconfirm

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✓ ビルドが完了しました！" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "出力ファイル: dist\KoeScribe-GUI.exe" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "注意事項:" -ForegroundColor Yellow
    Write-Host "  - 初回実行時、モデルファイルがダウンロードされます" -ForegroundColor Gray
    Write-Host "  - 話者分離機能を使用する場合、.envファイルが必要です" -ForegroundColor Gray
    Write-Host "  - .envファイルはexeと同じディレクトリに配置してください" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "✗ ビルドに失敗しました" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "エラーログを確認してください" -ForegroundColor Yellow
    exit 1
}

