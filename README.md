# whisper-local-server_GPU

Whisper を用いて日本語音声を文字起こしし、MeCab（UniDic辞書）で頻出名詞を解析するローカル処理ツールです。  
GPUによる高速処理に対応しています。

---

## 📦 主な機能

- 🎙 Whisper による日本語音声の文字起こし（mp4など対応）
- 📊 MeCab（UniDic）での頻出名詞ランキング出力
- ⏱ 処理時間の自動記録
- 📁 複数ファイルを `config.json` で一括処理
- ⚡ CUDA対応GPUでの高速化

---

## 🧱 ディレクトリ構成

```
whisper-local-server_GPUdesu/
├── 01.gpu_info.py                    # GPUとCUDAの確認
├── 02.run_transcription_whisper.py  # Whisperによる文字起こし
├── 03.analyze_word_frequency.py     # MeCabで名詞頻度を解析
├── config.json                      # 音声ファイルと出力先の設定
├── samples/Sample.m4a               # テスト用サンプル音声
├── result/                          # 処理結果出力ディレクトリ
├── Pipfile / Pipfile.lock           # pipenv 用の環境定義
```

---

## 🛠 セットアップ手順

### 1. リポジトリをクローン

```bash
git clone https://github.com/yourname/whisper-local-server_GPUdesu.git
cd whisper-local-server_GPUdesu
```

### 2. Python仮想環境構築（pipenv）

```bash
pipenv install
pipenv run python -m unidic download
```

### 3. ffmpeg をインストール（Whisper に必須）

- Windows: https://ffmpeg.org/download.html
- Mac: `brew install ffmpeg`

---

## 🧪 動作確認用スクリプト

### GPUとCUDAが有効かを確認

```bash
pipenv run python 01.gpu_info.py
```

### 出力例

```
torch_version = 2.3.0
cuda_available = True
cuda_device_count = 1
cuda_GPU_Num = 0
GPU_Name = NVIDIA GeForce RTX 3070 Ti
GPU_Compute_Capability = (8, 6)
```

---

## 🔍 GPUとCUDAバージョンの確認方法（Windows）

### NVIDIAドライバとCUDAの確認：

```bash
nvidia-smi
```

表示例：

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 560.94     Driver Version: 560.94     CUDA Version: 12.6        |
| GPU  Name              Memory-Usage   | GPU-Util | Temp   | Power           |
| 0    RTX 3070 Ti       5738MiB / 8GB  | 9%       | 53°C   | 50W / 290W      |
+-----------------------------------------------------------------------------+
```

### CUDA Toolkit バージョンを確認：

```bash
nvcc --version
```

出力例：

```
Cuda compilation tools, release 11.8, V11.8.89
```

---

## 🗂 config.jsonの例（サンプルファイル使用）

```json
{
  "root_dir": "./samples",
  "audio_files": ["Sample.m4a"],
  "output_dir": "./result"
}
```

---

## 🔁 使い方

### Whisperで音声文字起こし

```bash
pipenv run python 02.run_transcription_whisper.py
```

生成ファイル例：
- `result/output_sample/sample_transcription.txt`
- `result/output_sample/sample_processing_time.txt`

### MeCabで名詞頻度解析

```bash
pipenv run python 03.analyze_word_frequency.py
```

生成ファイル例：
- `result/output_sample/sample_word_frequency_mecab.txt`

---

## 📚 依存技術

- OpenAI Whisper
- PyTorch
- MeCab（mecab-python3）
- UniDic辞書
- ffmpeg, pipenv

---

## ✅ 推奨環境（2025年時点）

| 項目             | 推奨環境                    |
|------------------|-----------------------------|
| GPU              | NVIDIA GeForce RTX 30XX 以上 |
| CUDA Toolkit     | 11.8                        |
| PyTorch          | 2.3.0 (cu118)              |
| Python           | 3.12                        |
| OS               | Windows 10 / 11, WSL2可     |

---

## 📖 ライセンス

MIT License

