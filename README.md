# fwhisper-batch — Faster-Whisper batch transcriber with GUI

Faster-Whisper（CTranslate2版 Whisper）を **uv** で管理して、音声ファイルを一括で文字起こし・話者分離する統合システムです。  
Windows / macOS / Linux で動作します。PyTorch は不要です（CTranslate2 を使用）。

---

## ✅ 特長
- **高速・省メモリ**：CPU/ミドル級GPUで扱いやすい
- **日本語安定**：`language="ja"` 指定・VAD で長時間録音に強い
- **話者分離対応**：pyannote.audioで「誰がいつ何を話したか」を識別
- **GUI & CLI両対応**：使いやすいGUIアプリケーションとコマンドライン
- **エントリポイント**：`uv run fwhisper-batch` (CLI) / `uv run fwhisper-gui` (GUI)

---

## 📦 前提
- Python 3.9+
- **uv** インストール（https://docs.astral.sh/uv/）
- **ffmpeg** が PATH にあること  
  - macOS: `brew install ffmpeg` / Ubuntu: `sudo apt-get install -y ffmpeg`  
  - Windows: `winget install Gyan.FFmpeg` などで導入し、PATH を通す

---

## 🚀 セットアップ & 実行

### 1) 初回セットアップ
```bash
# プロジェクト直下（pyproject.toml がある場所）で
uv venv
uv sync
```

### 2) 設定ファイルの作成
```bash
cp config.json.example config.json
# エディタで root_dir / audio_files / output_dir などを編集
```

### 3) 話者分離機能の設定（オプション）
話者分離機能を使用する場合は、`.env`ファイルを作成してHugging Face トークンを設定：

```bash
# .envファイルを作成
echo "HUGGINGFACE_TOKEN=hf_your_token_here" > .env
```

**Hugging Face トークンの取得方法：**
1. https://huggingface.co でアカウント作成
2. Settings → Access Tokens → New token
3. Read権限のトークンを作成
4. 生成されたトークンを`.env`ファイルに設定

### 4) 実行方法

#### GUI アプリケーション（推奨）
```bash
uv run fwhisper-gui
```

#### コマンドライン
```bash
# 基本的な文字起こし
uv run fwhisper-batch --config config.json

# 話者分離付き（.envにHUGGINGFACE_TOKENが必要）
uv run fwhisper-batch --config config.json

# 話者分離を無効化
uv run fwhisper-batch --config config.json --disable-diarization
```

---

## 🖥️ GUI アプリケーション

### 主な機能
- **WAVファイル選択**：ドラッグ&ドロップまたはファイル選択
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

## 📱 実行ファイル版（ビルド済み）

### ビルド済み実行ファイルの使用
開発環境なしで使用したい場合は、ビルド済み実行ファイルを提供可能です。

**GPU設定について：**
- ビルド済み実行ファイルでも自動的にGPU検出
- NVIDIA GPU + 最新ドライバがあれば自動でGPU使用
- CUDA Toolkit等の追加インストール不要

### 実行ファイルのビルド方法
開発者向け：独自の実行ファイルを作成する場合

```bash
# ビルドスクリプトを実行
uv run python build_executable.py
```

**ビルド要件：**
- PyInstaller（自動インストール）
- 全依存関係がインストール済み
- config.json と .env ファイル

**出力：**
- `dist/FWhisper-GUI.exe`（Windows）
- 必要ファイル：config.json, .env（同じフォルダに配置）

---

## ⚙️ `config.json` の例
```json
{
  "root_dir": "./samples",
  "audio_files": ["sample1.wav", "sample2.mp3"],
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
  [fwhisper] model=large-v3 device=cuda compute_type=int8_float16
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
    uv run fwhisper-batch
    ```

### 動作確認ワンライナー
```powershell
uv run python -c "import ctranslate2 as c; print('CUDA GPUs:', c.get_cuda_device_count())"
```
`CUDA GPUs: 1` 以上なら GPU を掴めます（NVIDIA CUDA GPU 対応）。

---

## 🧰 CLI オプション
```bash
uv run fwhisper-batch --config config.json [オプション]
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
- `<basename>_segments.jsonl`：セグメント情報（時間・テキスト）
- `<basename>_words.jsonl`：単語レベル情報
- `<basename>_processing_time.txt`：処理統計情報

### 話者分離付き出力
- `<basename>_segments_with_speakers.jsonl`：話者情報付きセグメント
- `<basename>_words_with_speakers.jsonl`：話者情報付き単語
- `<basename>_segments_with_speakers.csv`：CSV形式（GUI自動生成）
- `<basename>_words_with_speakers.csv`：CSV形式（GUI自動生成）

### 出力例
```json
{"start": 0.0, "end": 3.2, "text": "おはようございます", "speaker": "SPEAKER_00"}
{"start": 3.5, "end": 6.8, "text": "今日の議題について", "speaker": "SPEAKER_00"}
{"start": 7.0, "end": 9.1, "text": "質問があります", "speaker": "SPEAKER_01"}
```

---

## 🧪 チューニングTips（日本語）
- 雑音/会議: `use_vad=true`、`min_silence_ms=700～1200` を試す
- 品質: `beam_size=8～10`（遅くなる）
- 速度: `beam_size=1～3`、量子化を強める（`int8`/`int8_float16`）

---

## ⚠️ トラブルシューティング

### 話者分離が動作しない
**症状**: 「完了 (話者分離スキップ)」と表示される
**解決策**:
1. `.env`ファイルに`HUGGINGFACE_TOKEN`が設定されているか確認
2. `config.json`に`diarize_model`が設定されているか確認
3. Hugging Face トークンが有効か確認

### GPU が認識されない
**症状**: CPU処理になってしまう
**解決策**:
```bash
# GPU確認コマンド
uv run python -c "import ctranslate2 as c; print('CUDA GPUs:', c.get_cuda_device_count())"
```
- NVIDIA ドライバを最新版に更新
- CUDA対応GPUか確認

### メモリ不足エラー
**症状**: CUDA out of memory
**解決策**:
- モデルサイズを`large-v3`→`medium`に変更
- `compute_type`を`int8`に変更

### その他のエラー
- **`fwhisper-batch: not found`**  
  → `uv sync` 前に実行していない/パッケージ化されていない可能性。  
  → プロジェクト直下で `uv sync` を実行。

- **`ModuleNotFoundError: No module named 'fwhisper_batch'`**  
  → フォルダ構成が正しいか確認し、`uv clean && uv sync`を実行。

- **ffmpeg が見つからない/読み込み失敗**  
  → OS へ ffmpeg をインストールし PATH を通す。

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

> NOTE: CTranslate2 の pip ホイールは必要な CUDA/cuDNN ランタイムを同梱しています。OS 側に CUDA Toolkit を入れていなくても動作します。

---

## 🔒 ライセンス
用途に応じて付与してください（例：MIT）。Faster-Whisper/CTranslate2 のライセンス準拠にご注意ください。

---

**fwhisper-batch** - 効率的な音声文字起こし・話者分離統合システム

