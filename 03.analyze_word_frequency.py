import MeCab
import unidic
from collections import Counter
from pathlib import Path
import os
import json

# config.json から設定を読み込み
with open("config.json", "r", encoding="utf-8") as cfg_file:
    config = json.load(cfg_file)

audio_files = config["audio_files"]
base_output_dir = config["output_dir"]

# 辞書パスを取得してスラッシュ形式に変換
dic_dir = Path(unidic.DICDIR).as_posix()
tagger = MeCab.Tagger(f"-d {dic_dir}")

# 各ファイルに対して名詞頻度を解析
for audio_filename in audio_files:
    basename = os.path.splitext(os.path.basename(audio_filename))[0]
    output_dir = os.path.join(base_output_dir, f"output_{basename}")
    transcription_path = os.path.join(output_dir, f"{basename}_transcription.txt")

    if not os.path.exists(transcription_path):
        print(f"⚠️ {transcription_path} が見つかりません。スキップします。")
        continue

    with open(transcription_path, "r", encoding="utf-8") as f:
        text = f.read()

    node = tagger.parseToNode(text)
    nouns = []

    while node:
        features = node.feature.split(",")
        if features[0] == "名詞":
            surface = node.surface
            if surface:
                nouns.append(surface)
        node = node.next

    counter = Counter(nouns)

    output_path = os.path.join(output_dir, f"{basename}_word_frequency_mecab.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        print(f"\n=== {basename} の上位頻出名詞ランキング ===")
        for word, count in counter.most_common(50):
            line = f"{word}: {count}回"
            print(line)
            f.write(line + "\n")

