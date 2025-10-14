# KoeScribe-音声文字起こし・話者分離 GUI アプリケーション

Faster-Whisper と pyannote.audio を使用した、高精度な音声文字起こしと話者分離を行うデスクトップアプリケーションです。  
**Windows / macOS / Linux** 対応、直感的なGUIで複数ファイルの一括処理が可能です。

---

## ✨ 主な機能

- 🎯 **高精度文字起こし**: Faster-Whisper による高速・高精度な音声認識
- 👥 **話者分離**: 誰がいつ話したかを自動識別（pyannote.audio）
- 📊 **リアルタイム進捗**: 処理状況と経過時間をリアルタイム表示
- 🔄 **キュー処理**: 複数ファイルの連続処理に対応
- 📈 **CSV出力**: 結果をCSV形式でダウンロード可能
- 💾 **設定管理**: プロジェクトごとの設定保存・管理
- 🎛️ **詳細設定**: 話者数、モデルサイズ、出力形式などを柔軟に設定

---

## 🔧 システム要件

- **Python**: 3.9以上（3.10-3.11推奨）
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **メモリ**: 8GB以上推奨（4GBでも動作可能）
- **GPU**: NVIDIA GPU推奨（CPUでも動作）

---

## 📦 インストール

### 1. 前提条件のインストール

#### Windows
```powershell
# uvをインストール
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

```

#### macOS
```bash
# uvをインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

```

#### Linux (Ubuntu/Debian)
```bash
# uvをインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

```

### 2. プロジェクトのセットアップ

```bash
# リポジトリをクローン
git clone https://github.com/sassa4771/whisper-local-server_CPU.git
cd whisper-local-server_CPU

# 依存関係をインストール
uv sync
```

**注意**: 以前のバージョンで `webrtcvad` のビルドエラーが発生していた場合、最新版では依存関係から削除されているため、エラーは解消されています。

### 3. 話者分離機能の設定（オプション）

話者分離機能を使用する場合は、Hugging Face トークンが必要です：

#### Hugging Face トークンの取得
1. [Hugging Face](https://huggingface.co) でアカウントを作成
2. [Settings > Access Tokens](https://huggingface.co/settings/tokens) でアクセストークンを作成
3. **Read** 権限のトークンを生成

#### 設定ファイルの作成
プロジェクトルートに `.env` ファイルを作成：

```bash
# .envファイルを作成
echo "HUGGINGFACE_TOKEN=hf_your_token_here" > .env
```

**Windows PowerShell の場合:**
```powershell
echo "HUGGINGFACE_TOKEN=hf_your_token_here" | Out-File -FilePath .env -Encoding utf8
```

> [!IMPORTANT]
> .envファイルを作成して、
> hf_your_token_hereに自分のトークンを設定してください。

---

## 🚀 使用方法

### GUI アプリケーション（推奨）

```bash
uv run koescribe-gui
```

#### 基本的な操作手順
1. **ファイル選択**: 「ファイル追加」または「フォルダ追加」で音声・動画ファイルを選択
2. **設定調整**: 話者数、出力ディレクトリ、モデルサイズなどを設定
3. **処理開始**: 「処理開始」ボタンをクリック
4. **進捗確認**: リアルタイムで処理状況と経過時間を確認
5. **結果確認**: 処理完了後、結果テーブルから出力ディレクトリにアクセス
6. **CSV出力**: 必要に応じてCSV形式で結果をダウンロード

#### 対応ファイル形式
- **音声**: WAV

### コマンドライン

```bash
# 基本的な文字起こし
uv run koescribe-batch --config config.json --files audio.wav

# 話者分離付き（.envにHUGGINGFACE_TOKENが必要）
uv run koescribe-batch --config config.json --files audio.wav

# 話者分離を無効化
uv run koescribe-batch --config config.json --files audio.wav --disable-diarization

```

---

## 🖥️ GUI アプリケーション

### 主な機能
- **動的キュー管理**：処理中でもファイル追加可能
- **設定プリセット**：よく使う設定を保存・読み込み
- **リアルタイム進捗**：処理状況をリアルタイム表示
- **結果管理**：処理履歴の保存・CSV出力
- **ヘルプシステム**：各設定項目の詳細説明

### 設定項目
- **モデルサイズ**：large-v3（高精度）/ medium（バランス）/ small（高速）
- **言語**：ja（日本語）/ en（英語）/ auto（自動検出）
- **デバイス**：auto（自動）/ cuda（GPU）/ cpu（CPU）
- **話者分離**：有効/無効（HUGGINGFACE_TOKEN必須）
- **最大話者数**：2-10人（デフォルト：2）

---


## ⚙️ `config.json` の例
```json
{
  "root_dir": "./samples",
  "audio_files": ["sample1.wav", "sample2.wav"],
  "output_dir": "./outputs",
  "model_size": "large-v3",
  "device": "auto",
  "compute_type": "auto",
  "beam_size": 5,
  "use_vad": true,
  "min_silence_ms": 500,
  "language": "ja",
  
  "diarize_model": "pyannote/speaker-diarization",
  "diarize_min_dur": 0.8,
  "diarize_bridge_gap": 0.3
}
```
- **root_dir**: 音声ファイルのルートディレクトリ
- **audio_files**: 相対/絶対パスどちらでも可（配列）
- **output_dir**: 出力先ルート
- **model_size**: `large-v3` / `medium` / `small` / `distil-large-v3` など
- **device**: `auto` | `cuda` | `cpu`
- **compute_type**: `auto` | `int8_float16`（GPU向け）| `int8`（CPU向け）| `float16` など
- **beam_size**: 1～10（大きいほど精度↑・速度↓）
- **use_vad**: 無音検出の有無（長時間録音や会議に有効）
- **min_silence_ms**: 区間切りの無音長（例: 500～1200ms）
- **language**: 既知なら `"ja"` 固定が安定

---

## ⚡ GPU / CPU の自動判定と出力ログ
- 既定（`"device": "auto"`）では **CUDA GPU が見つかれば `cuda`、無ければ `cpu`** を自動選択します。
- 既定（`"compute_type": "auto"`）では **`cuda` 時は `int8_float16`、`cpu` 時は `int8`** を自動選択します。
- 起動時に次のようなログが出ます：
  ```text
  [koescribe] model=large-v3 device=cuda compute_type=int8_float16
  ```
  `device=cuda` なら **GPU 使用中**、`device=cpu` なら **CPU 使用中** です。

### 手動で固定したい場合
- `config.json` で指定：
  ```json
  "device": "cuda",       // または "cpu"
  "compute_type": "int8_float16"
  ```
- 一時的に環境変数で：
  - PowerShell:
    ```powershell
    $env:FWHISPER_DEVICE="cuda"
    $env:FWHISPER_COMPUTE="int8_float16"
    uv run koescribe-batch
    ```

### 動作確認ワンライナー
```powershell
uv run python -c "import ctranslate2 as c; print('CUDA GPUs:', c.get_cuda_device_count())"
```
`CUDA GPUs: 1` 以上なら GPU を掴めます（NVIDIA CUDA GPU 対応）。

---

## 🧰 CLI オプション
```bash
uv run koescribe-batch --config config.json [オプション]
```

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--config` | 設定ファイルパス | `config.json` |
| `--files` | 音声ファイルリスト | config.jsonから取得 |
| `--root` | ルートディレクトリ | config.jsonから取得 |
| `--out` | 出力ディレクトリ | config.jsonから取得 |
| `--word-timestamps` | 単語レベルタイムスタンプ強制有効 | False |
| `--disable-diarization` | 話者分離無効化 | False |
| `--no-progress` | プログレスバー非表示 | False |

---

## 📂 出力ファイル

### 基本出力（文字起こしのみ）
- `<basename>_segments.csv`：セグメント情報（時間・テキスト）
- `<basename>_words.csv`：単語レベル情報
- `<basename>_processing_time.txt`：処理統計情報

### 話者分離付き出力
- `<basename>_segments_with_speakers.csv`：話者情報付きセグメント
- `<basename>_words_with_speakers.csv`：話者情報付き単語

### 出力例（CSV形式）
```csv
start,end,text,speaker
0.0,3.2,おはようございます,SPEAKER_00
3.5,6.8,今日の議題について,SPEAKER_00
7.0,9.1,質問があります,SPEAKER_01
```

---

## 🧪 チューニングTips（日本語）
- 雑音/会議: `use_vad=true`、`min_silence_ms=700～1200` を試す
- 品質: `beam_size=8～10`（遅くなる）
- 速度: `beam_size=1～3`、量子化を強める（`int8`/`int8_float16`）

---

## ⚠️ トラブルシューティング

### インストール関連

#### 1. 依存関係のインストールエラー
**症状**: `uv sync` でエラーが発生
**解決策**:
```bash
# キャッシュをクリアして再インストール
uv clean
uv sync

# Python バージョンを確認
python --version  # 3.9以上が必要
```

### 話者分離関連

#### 2. 話者分離が動作しない
**症状**: 「完了 (話者分離スキップ)」と表示される
**解決策**:
1. `.env`ファイルに`HUGGINGFACE_TOKEN`が設定されているか確認
2. `config.json`に`diarize_model`が設定されているか確認
3. Hugging Face トークンが有効か確認
4. pyannote/speaker-diarization モデルの利用規約に同意しているか確認

#### 3. Hugging Face トークンエラー
**症状**: `Authentication failed` エラー
**解決策**:
```bash
# トークンの確認
cat .env

# 正しい形式: HUGGINGFACE_TOKEN=hf_xxxxxxxxxx
# hf_ で始まる必要があります
```

### GPU・パフォーマンス関連

#### 5. GPU が認識されない
**症状**: CPU処理になってしまう
**解決策**:
```bash
# GPU確認コマンド
uv run python -c "import ctranslate2 as c; print('CUDA GPUs:', c.get_cuda_device_count())"
```
- NVIDIA ドライバを最新版に更新
- CUDA対応GPUか確認（GTX 10シリーズ以降推奨）

#### 6. メモリ不足エラー
**症状**: `CUDA out of memory` または `RuntimeError: out of memory`
**解決策**:
- モデルサイズを変更: `large-v3` → `medium` → `small`
- `compute_type`を`int8`に変更
- 他のアプリケーションを終了してメモリを確保

#### 7. 処理が遅い
**症状**: 処理時間が長すぎる
**解決策**:
- GPU使用を確認（上記GPU確認コマンド）
- モデルサイズを小さくする
- `beam_size`を小さくする（5 → 3 → 1）

### アプリケーション関連

#### 8. GUI が起動しない
**症状**: `uv run koescribe-gui` でエラー
**解決策**:
```bash
# 詳細エラーを確認
uv run koescribe-gui --debug

# 依存関係を再インストール
uv sync --reinstall
```

#### 9. ファイルが読み込めない
**症状**: 音声・動画ファイルが処理できない
**解決策**:
- ファイル形式を確認（WAVのみ対応）
- ファイルパスに日本語や特殊文字が含まれていないか確認
- ファイルが破損していないか確認

#### 10. 設定が保存されない
**症状**: アプリを再起動すると設定がリセットされる
**解決策**:
- アプリケーションに書き込み権限があるか確認
- ウイルス対策ソフトがブロックしていないか確認

### その他のエラー

#### 11. コマンドが見つからない
**症状**: `koescribe-batch: not found` または `koescribe-gui: not found`
**解決策**:
```bash
# プロジェクトディレクトリで実行しているか確認
pwd
ls pyproject.toml  # このファイルがあることを確認

# 依存関係を再インストール
uv sync
```

#### 12. モジュールが見つからない
**症状**: `ModuleNotFoundError: No module named 'fwhisper_batch'`
**解決策**:
```bash
# フォルダ構成を確認
ls src/fwhisper_batch/

# 完全にクリーンインストール
uv clean
rm -rf .venv  # 仮想環境を削除
uv sync
```

### ログの確認方法

詳細なエラー情報を確認したい場合：

```bash
# GUIアプリケーションをデバッグモードで起動
uv run koescribe-gui --debug

# コマンドラインで詳細ログを表示
uv run koescribe-batch --config config.json --files audio.wav --verbose
```

## 📊 パフォーマンス目安

### CPU処理時（Intel i7, 16GB RAM）
| 音声長 | モデル | 処理時間 | 話者分離 |
|--------|--------|----------|----------|
| 10分 | medium | ~2分 | +30秒 |
| 30分 | large-v3 | ~8分 | +1分 |
| 60分 | large-v3 | ~15分 | +2分 |

### GPU処理時（RTX 3080）
| 音声長 | モデル | 処理時間 | 話者分離 |
|--------|--------|----------|----------|
| 10分 | large-v3 | ~30秒 | +15秒 |
| 30分 | large-v3 | ~1.5分 | +30秒 |
| 60分 | large-v3 | ~3分 | +45秒 |

---

## 🧩 GPU / CUDA のセットアップは必要？

**結論：必須ではありません。**  
Faster-Whisper は CTranslate2 の **GPU 対応ホイール（pip）** を利用しており、**CUDA Toolkit や cuDNN を別途インストールしなくても動作**します。  
必要なのは **対応する NVIDIA GPU＋適切なドライバ** だけです（インストール済みのドライバが最近のものであればOK）。

### GPU 利用可否の確認
```bash
uv run python -c "import ctranslate2 as c; print('CUDA GPUs:', c.get_cuda_device_count())"
```
`CUDA GPUs: 1` 以上が表示されれば、`device=auto` で **GPU (cuda)** が選ばれます。

> NOTE: CTranslate2 の pip ホイールは必要な CUDA/cuDNN ランタイムを同梱しています。OS 側に CUDA Toolkit を入れ

---

