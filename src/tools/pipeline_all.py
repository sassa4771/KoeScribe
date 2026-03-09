import os, json, argparse, time
from pathlib import Path
from dotenv import load_dotenv

from faster_whisper import WhisperModel
import ctranslate2
import pandas as pd

# 既存ライブラリ関数を利用
from fwhisper_batch.transcribe_batch import resolve_device, resolve_compute_type
from tools.diarize import diarize_one
from tools.merge_speakers import assign_speakers_to_words, to_runs

def write_csv(path: Path, rows):
    """
    CSVファイルを書き込む。権限エラーなどの問題を適切に処理する。
    """
    try:
        # ディレクトリを作成
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if rows:
            df = pd.DataFrame(rows)
            
            # 既存のファイルが読み取り専用の場合、属性を変更してから削除を試みる
            if path.exists():
                try:
                    # Windowsで読み取り専用属性を解除
                    import os
                    if os.name == 'nt':  # Windows
                        import stat
                        current_attrs = path.stat().st_file_attributes
                        if current_attrs & stat.FILE_ATTRIBUTE_READONLY:
                            path.chmod(stat.S_IWRITE)
                except Exception:
                    pass  # 属性変更に失敗しても続行
            
            # 一時ファイルに書き込んでからリネーム（アトミック書き込み）
            temp_path = path.with_suffix('.tmp')
            try:
                df.to_csv(temp_path, index=False, encoding='utf-8-sig')
                # 既存ファイルがあれば削除
                if path.exists():
                    path.unlink()
                # 一時ファイルをリネーム
                temp_path.rename(path)
            except PermissionError as e:
                # 一時ファイルをクリーンアップ
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass
                raise PermissionError(
                    f"ファイル '{path}' への書き込み権限がありません。\n"
                    f"考えられる原因:\n"
                    f"  1. ファイルが他のプログラム（Excel、テキストエディタなど）で開かれています\n"
                    f"  2. ファイルが読み取り専用になっています\n"
                    f"  3. ディレクトリへの書き込み権限がありません\n"
                    f"  4. ウイルス対策ソフトがブロックしています\n"
                    f"\n解決方法:\n"
                    f"  - ファイルを開いているプログラムをすべて閉じてください\n"
                    f"  - ファイルのプロパティで読み取り専用を解除してください\n"
                    f"  - 管理者権限で実行してみてください\n"
                    f"  元のエラー: {str(e)}"
                )
    except PermissionError:
        raise  # 上で処理したPermissionErrorを再発生
    except Exception as e:
        raise Exception(
            f"CSVファイル '{path}' の書き込みに失敗しました: {str(e)}\n"
            f"ファイルパス: {path}\n"
            f"ディレクトリの書き込み権限を確認してください。"
        )

def asr_words(model: WhisperModel, audio_path: Path, language: str, beam_size: int):
    segs, info = model.transcribe(
        str(audio_path),
        language=language or None,
        beam_size=beam_size,
        vad_filter=True,
        word_timestamps=True
    )
    words=[]
    for s in segs:
        if getattr(s, "words", None):
            for w in s.words:
                words.append({"start": w.start, "end": w.end, "word": w.word})
    return words

def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="ASR+DIARIZE+MERGE one-shot")
    ap.add_argument("--audio", nargs="+", required=True)
    ap.add_argument("--out-dir", default="./outputs")
    ap.add_argument("--model-size", default="large-v3")
    ap.add_argument("--language", default="ja")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--diarize-model", default="pyannote/speaker-diarization-3.1")
    ap.add_argument("--min-dur", type=float, default=0.8)
    ap.add_argument("--bridge-gap", type=float, default=0.3)
    args = ap.parse_args()

    device = resolve_device(os.getenv("FWHISPER_DEVICE","auto"))
    compute = resolve_compute_type(device, os.getenv("FWHISPER_COMPUTE","auto"))
    print(f"[pipeline] model={args.model_size} device={device} compute={compute}")
    model = WhisperModel(args.model_size, device=device, compute_type=compute)

    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit("HUGGINGFACE_TOKEN が未設定です（.env に記載してください）。")

    out_root = Path(args.out_dir)

    for apath in map(Path, args.audio):
        out_dir = out_root / f"output_{apath.stem}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) ASR (words)
        t0=time.time()
        words = asr_words(model, apath, args.language, args.beam_size)
        write_csv(out_dir / f"{apath.stem}_words.csv", words)
        print(f"[asr] words: {len(words)}")

        # 2) Diarize
        spans_path = out_dir / "spans.csv"
        diarize_one(apath, spans_path, args.diarize_model, token, args.min_dur, args.bridge_gap)
        df = pd.read_csv(spans_path, encoding='utf-8-sig')
        spans = df.to_dict('records')

        # 3) Merge
        labeled = assign_speakers_to_words(words, spans, smooth_min_sec=0.6)
        write_csv(out_dir / f"{apath.stem}_merged.csv", labeled)
        runs = to_runs(labeled)
        with (out_dir / f"{apath.stem}_transcription.txt").open("w", encoding="utf-8") as f:
            for r in runs:
                f.write(f"[{r['speaker']}] {r['text'].strip()}\n")

        print(f"[done] {apath.name} in {time.time()-t0:.1f}s → {out_dir}")

if __name__ == "__main__":
    main()
