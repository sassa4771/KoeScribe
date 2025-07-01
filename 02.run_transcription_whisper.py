import whisper
import time
from janome.tokenizer import Tokenizer
import os
import json
from tqdm import tqdm

# JSON設定ファイル読み込み
with open("config.json", "r", encoding="utf-8") as cfg_file:
    config = json.load(cfg_file)

root_dir = config["root_dir"]
audio_files = config["audio_files"]
base_output_dir = config["output_dir"]

# Whisperモデルの読み込み
model = whisper.load_model("large")
tokenizer = Tokenizer()

# 指定された全ファイルを処理
for idx, audio_filename in enumerate(tqdm(audio_files, desc="🎧 Transcribing", unit="file"), start=1):
    audio_path = os.path.join(root_dir, audio_filename)
    basename = os.path.splitext(os.path.basename(audio_filename))[0]

    output_dir = os.path.join(base_output_dir, f"output_{basename}")
    os.makedirs(output_dir, exist_ok=True)

    # 文字起こし処理
    start = time.time()
    result = model.transcribe(audio_path, language="ja", fp16=False)
    end = time.time()

    text = result["text"]
    processing_time = end - start

    # 結果の保存
    transcription_file = os.path.join(output_dir, f"{basename}_transcription.txt")
    with open(transcription_file, "w", encoding="utf-8") as f:
        f.write(text)

    time_file = os.path.join(output_dir, f"{basename}_processing_time.txt")
    with open(time_file, "w", encoding="utf-8") as f:
        f.write(f"処理時間: {processing_time:.2f} 秒\n")
        f.write(f"実行デバイス: {model.device}\n")

    # 結果表示
    print(f"\n=== Transcription Result: {basename} ===")
    print(text)
    print(f"\n⏱️ 処理時間: {processing_time:.2f} 秒")
    print(f"✅ 実行デバイス: {model.device}")

    # ✅ メール通知（各ファイルごと）
    subject = f"✅ {idx}/{len(audio_files)} 完了: {basename}"
    body = (
        f"{basename} の文字起こしが完了しました。\n"
        f"処理時間: {processing_time:.2f} 秒\n"
        f"実行デバイス: {model.device}"
    )
    send_email_notification(subject, body)

print("\n✅ 全ての処理が完了しました。")
