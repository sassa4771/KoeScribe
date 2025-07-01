# whisper-local-server_CPU

Whisper を用いて日本語音声を文字起こしするローカル処理ツールです。  

---

## 📦 主な機能

- 🎙 Whisper による日本語音声の文字起こし（mp4など対応）
- ⏱ 処理時間の自動記録
- 📁 複数ファイルを `config.json` で一括処理

---

## 🧱 ディレクトリ構成

```
whisper-local-server_GPUdesu/
├── run_transcription_whisper.py     # Whisperによる文字起こし
├── config.json                      # 音声ファイルと出力先の設定
├── samples/Sample.m4a               # テスト用サンプル音声
├── result/                          # 処理結果出力ディレクトリ
├── Pipfile / Pipfile.lock           # pipenv 用の環境定義
```

---

## 🛠 セットアップ手順

### 1. リポジトリをクローン

```bash
git clone https://github.com/IPTeCA/whisper-local-server_CPU.git
cd whisper-local-server_CPU
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
pipenv run python run_transcription_whisper.py
```

生成ファイル例：
- `result/output_sample/Sample_transcription.txt`
- `result/output_sample/Sample_processing_time.txt`


## 📚 依存技術

- OpenAI Whisper
- PyTorch
- ffmpeg, pipenv

---

## ✅ 推奨環境（2025年時点）

| 項目    | 推奨環境例                                                     |
|---------|----------------------------------------------------------------|
| CPU     | 8 コア以上／AVX2 対応 3 GHz 以上<br>例：Intel Core i7-12700、AMD Ryzen 7 5800X |
| Memory  | 16 GB 以上（Whisper-large を扱うなら 32 GB 推奨）              |
| PyTorch | 2.3.0 **CPU ビルド**<br>↳ `pip install torch==2.3.0+cpu`        |
| Python  | 3.12（3.10 以上であれば可）                                    |
| OS      | Windows 10/11（WSL2 可）／Linux／macOS                         |

---

## 📖 ライセンス

MIT License

