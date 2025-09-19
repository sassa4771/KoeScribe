# fwhisper-batch — Faster-Whisper batch transcriber (uv)

Faster-Whisper（CTranslate2版 Whisper）を **uv** で管理して、音声ファイルを一括で文字起こしする最小プロジェクトです。  
Windows / macOS / Linux で動作します。PyTorch は不要です（CTranslate2 を使用）。

---

## ✅ 特長
- **高速・省メモリ**：CPU/ミドル級GPUで扱いやすい
- **日本語安定**：`language="ja"` 指定・VAD で長時間録音に強い
- **シンプル出力**：TXT / 処理時間（統計）/ Segments JSONL
- **エントリポイント**：`uv run fwhisper-batch` で起動（console script）

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

### 3) 実行（推奨：エントリポイント）
```bash
uv run fwhisper-batch --config config.json
```

#### 代替：直接モジュール/ファイルで実行
```bash
uv run python -m fwhisper_batch.transcribe_batch --config config.json
# または
uv run python ./src/fwhisper_batch/transcribe_batch.py --config config.json
```

> Windows PowerShell でも同様に動作します。

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
  "language": "ja"
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
uv run fwhisper-batch --config config.json [--word-timestamps] [--no-progress]
uv run fwhisper-batch --root ./data --out ./out --files a.wav b.mp3
```
- `--config`: 設定ファイルのパス（既定: `config.json`）
- `--root`: `root_dir` を上書き
- `--out`: `output_dir` を上書き
- `--files`: `audio_files` を上書き（空白区切り）
- `--word-timestamps`: 単語タイムスタンプを `segments.jsonl` に含める
- `--no-progress`: ファイル内進捗バーを非表示

---

## 📂 出力
各ファイルごとに `outputs/output_<basename>/` を作成：
- `<basename>_transcription.txt` … 文字起こし本文（UTF-8）
- `<basename>_processing_time.txt` … 処理時間や推定言語、音声長の統計
- `<basename>_segments.jsonl` … 区間ごとの `{start,end,text,words?}`（1行1JSON）

> SRT/WebVTT への変換は簡単に拡張可能です（必要ならスニペット提供します）。

---

## 🧪 チューニングTips（日本語）
- 雑音/会議: `use_vad=true`、`min_silence_ms=700～1200` を試す
- 品質: `beam_size=8～10`（遅くなる）
- 速度: `beam_size=1～3`、量子化を強める（`int8`/`int8_float16`）

---

## 🛠 トラブルシュート
- **`fwhisper-batch: not found`**  
  → `uv sync` 前に実行していない/パッケージ化されていない可能性。  
  → プロジェクト直下で `uv sync` を実行。`pyproject.toml` の
  ```toml
  [project.scripts]
  fwhisper-batch = "fwhisper_batch.transcribe_batch:main"

  [tool.uv]
  package = true

  [tool.uv.sources]
  fwhisper_batch = { path = "src/fwhisper_batch" }
  ```
  を確認。

- **`ModuleNotFoundError: No module named 'fwhisper_batch'`**  
  → フォルダ構成が正しいか確認：
  ```text
  project-root/
    pyproject.toml
    src/
      fwhisper_batch/
        __init__.py
        transcribe_batch.py   # ← main() が定義されている
  ```
  → 直した後は `uv clean && uv sync`。

- **ffmpeg が見つからない/読み込み失敗**  
  → OS へ ffmpeg をインストールし PATH を通す。

---

## 📁 プロジェクト構成（推奨）
```text
fwhisper-uv/
  pyproject.toml
  README.md
  config.json.example
  src/
    fwhisper_batch/
      __init__.py
      transcribe_batch.py
  outputs/        # 実行時に作成されます
  samples/        # 任意
```

---

## 🔒 ライセンス
用途に応じて付与してください（例：MIT）。Faster-Whisper/CTranslate2 のライセンス準拠にご注意ください。

---

## ❓サポート
設定や pyannote 連携（話者分離→話者ラベル付き文字起こし）、SRT/WebVTT 出力を追加したい場合は声をかけてください。


---

## 🧩 GPU / CUDA のセットアップは必要？

**結論：必須ではありません。**  
Faster-Whisper は CTranslate2 の **GPU 対応ホイール（pip）** を利用しており、**CUDA Toolkit や cuDNN を別途インストールしなくても動作**します。  
必要なのは **対応する NVIDIA GPU＋適切なドライバ** だけです（インストール済みのドライバが最近のものであればOK）。

> 例外：既に CUDA Toolkit を導入済みでも問題ありません（共存可）。また、PyTorch を別用途（例：pyannote）で使う場合は、そちらの推奨に従って CUDA/cuDNN を用意してください。

### GPU 利用可否の確認
```powershell
uv run python -c "import ctranslate2 as c; print('CUDA GPUs:', c.get_cuda_device_count())"
```
`CUDA GPUs: 1` 以上が表示されれば、`device=auto` で **GPU (cuda)** が選ばれます。

> NOTE: CTranslate2 の pip ホイールは必要な CUDA/cuDNN ランタイムを同梱しています。OS 側に CUDA Toolkit を入れていなくても動作します。

---

## ✅ 推奨環境（2025年時点・動作確認例）

> 下表は実機での **動作確認済み例** です。Faster-Whisper 単体では PyTorch は不要ですが、**pyannote 等の連携用途**や他の研究環境と合わせる場合の参考構成として記載します。

| 項目           | 推奨/確認済み環境                                   |
| -------------- | --------------------------------------------------- |
| GPU            | NVIDIA GeForce **RTX 30XX 以上**                    |
| NVIDIA Driver  | 最新版推奨（CUDA 11.8 対応相当以上）                 |
| CUDA Toolkit   | **任意**（未導入でも可／導入するなら **11.8** 推奨） |
| cuDNN          | **任意**（pip ホイール同梱。導入するなら **9.x**）   |
| PyTorch        | **任意**（連携用途例：**2.3.0 (cu118)**）           |
| Python         | **3.12**                                            |
| OS             | **Windows 10 / 11**、または **WSL2 (Ubuntu)**       |

> 補足：上記は「GPU 利用の安定運用」を意識した構成です。CPU だけで使う場合は GPU 関連の準備は不要です。

---

