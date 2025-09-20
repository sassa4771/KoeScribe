# fwhisper-batch 統合バッチ処理システム

音声ファイルの文字起こしと話者分離を一括で実行する統合システムです。Faster-Whisperとpyannote.audioを使用して、「誰がいつ何を話したか」を効率的に識別できます。

## 🚀 特徴

- **一括処理**: 複数の音声ファイルを一度に処理
- **自動話者分離**: 設定に基づいて自動的に話者を識別
- **高速処理**: CPU/GPU対応のFaster-Whisperエンジン
- **柔軟な出力**: セグメント・単語レベルでの話者情報付きJSONL出力
- **エラー耐性**: 話者分離失敗時も通常の文字起こしを継続

## 📋 必要な環境

### システム要件
- Python 3.8+
- uv (パッケージマネージャー)
- CUDA対応GPU（オプション、高速化のため）

### 依存関係
```bash
# プロジェクトのセットアップ
uv sync
```

## ⚙️ 設定

### 1. 設定ファイル (config.json)

```json
{
  "root_dir": "./audio_files",
  "audio_files": ["meeting.wav", "interview.m4a"],
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

### 2. 環境変数 (.env)

話者分離機能を使用する場合は必須：

```bash
# Hugging Face トークン（pyannote.audio用）
HUGGINGFACE_TOKEN=hf_your_token_here

# オプション: デバイス強制指定
# FWHISPER_DEVICE=cuda
# FWHISPER_COMPUTE=int8_float16
```

## 🎯 使用方法

### 基本コマンド

```bash
# 設定ファイルを使用した一括処理（話者分離自動有効化）
uv run fwhisper-batch --config config.json

# 特定ファイルを直接指定
uv run fwhisper-batch --config config.json --files audio1.wav audio2.m4a

# 話者分離を無効化
uv run fwhisper-batch --config config.json --disable-diarization

# 出力ディレクトリを指定
uv run fwhisper-batch --config config.json --out ./custom_output
```

### オプション一覧

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--config` | 設定ファイルパス | `config.json` |
| `--files` | 音声ファイルリスト | config.jsonから取得 |
| `--root` | ルートディレクトリ | config.jsonから取得 |
| `--out` | 出力ディレクトリ | config.jsonから取得 |
| `--word-timestamps` | 単語レベルタイムスタンプ強制有効 | False |
| `--disable-diarization` | 話者分離無効化 | False |
| `--no-progress` | プログレスバー非表示 | False |

## 📁 出力ファイル

各音声ファイル（例: `meeting.wav`）に対して以下が生成されます：

### 基本出力
- `meeting_segments.jsonl`: セグメント情報（時間・テキスト）
- `meeting_words.jsonl`: 単語レベル情報（話者分離時は自動生成）
- `meeting_processing_time.txt`: 処理統計情報

### 話者分離付き出力
- `meeting_segments_with_speakers.jsonl`: 話者情報付きセグメント
- `meeting_words_with_speakers.jsonl`: 話者情報付き単語

### 出力例

**segments_with_speakers.jsonl:**
```json
{"start": 0.0, "end": 3.2, "text": "おはようございます", "speaker": "SPEAKER_00"}
{"start": 3.5, "end": 6.8, "text": "今日の議題について", "speaker": "SPEAKER_00"}
{"start": 7.0, "end": 9.1, "text": "質問があります", "speaker": "SPEAKER_01"}
```

**words_with_speakers.jsonl:**
```json
{"start": 0.0, "end": 0.8, "word": "おはよう", "speaker": "SPEAKER_00"}
{"start": 0.8, "end": 1.5, "word": "ございます", "speaker": "SPEAKER_00"}
{"start": 7.0, "end": 7.4, "word": "質問", "speaker": "SPEAKER_01"}
```

## 🔧 実用例

### 会議録音の処理

```bash
# 会議用設定ファイル作成
cat > meeting_config.json << EOF
{
  "root_dir": "./meetings",
  "audio_files": ["team_meeting_20240920.wav"],
  "output_dir": "./meeting_outputs",
  "model_size": "large-v3",
  "language": "ja",
  "diarize_model": "pyannote/speaker-diarization",
  "diarize_min_dur": 1.0,
  "diarize_bridge_gap": 0.5
}
EOF

# 実行
uv run fwhisper-batch --config meeting_config.json
```

### 複数ファイル一括処理

```bash
# 複数の音声ファイルを一度に処理
uv run fwhisper-batch --config config.json \
  --files interview1.wav interview2.m4a lecture.mp3 \
  --out ./batch_results
```

### 高速処理（GPU使用）

```bash
# GPU設定を明示的に指定
FWHISPER_DEVICE=cuda FWHISPER_COMPUTE=int8_float16 \
uv run fwhisper-batch --config config.json
```

## ⚠️ トラブルシューティング

### 話者分離が動作しない

**症状**: 話者情報が付与されない
**解決策**:
1. `.env`ファイルに`HUGGINGFACE_TOKEN`が設定されているか確認
2. `config.json`に`diarize_model`が設定されているか確認
3. 音声ファイル形式がサポートされているか確認（wav, m4a, mp3など）

### メモリ不足エラー

**症状**: CUDA out of memory
**解決策**:
```json
{
  "model_size": "medium",  // large-v3 → medium に変更
  "compute_type": "int8"   // メモリ使用量削減
}
```

### 音声ファイルが見つからない

**症状**: "Input files not found"
**解決策**:
1. `root_dir`のパスが正しいか確認
2. 音声ファイル名が正確か確認
3. 相対パス/絶対パスの設定を確認

## 🔄 処理フロー

1. **設定読み込み**: config.jsonと.envから設定を取得
2. **話者分離判定**: diarize_modelとHUGGINGFACE_TOKENの存在確認
3. **モデル初期化**: Faster-Whisperモデルをロード
4. **ファイル処理**: 各音声ファイルに対して
   - 文字起こし実行
   - 話者分離実行（有効な場合）
   - 結果マージ・保存
5. **結果出力**: JSONL形式で保存

## 📊 パフォーマンス

### 処理速度目安（CPU: Intel i7, 16GB RAM）

| 音声長 | モデル | 処理時間 | 話者分離 |
|--------|--------|----------|----------|
| 10分 | medium | ~2分 | +30秒 |
| 30分 | large-v3 | ~8分 | +1分 |
| 60分 | large-v3 | ~15分 | +2分 |

### GPU使用時（RTX 3080）

| 音声長 | モデル | 処理時間 | 話者分離 |
|--------|--------|----------|----------|
| 10分 | large-v3 | ~30秒 | +15秒 |
| 30分 | large-v3 | ~1.5分 | +30秒 |
| 60分 | large-v3 | ~3分 | +45秒 |

## 🤝 サポート

問題が発生した場合は、以下の情報と共にお問い合わせください：

1. 使用したコマンド
2. config.jsonの内容
3. エラーメッセージ
4. 音声ファイルの形式・長さ
5. システム環境（OS、Python版、GPU有無）

---

**fwhisper-batch** - 効率的な音声文字起こし・話者分離統合システム
