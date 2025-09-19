import os, json, argparse, time
from pathlib import Path
from dotenv import load_dotenv

from faster_whisper import WhisperModel
import ctranslate2

# 既存ライブラリ関数を利用
from fwhisper_batch.transcribe_batch import resolve_device, resolve_compute_type
from tools.diarize import diarize_one
from tools.merge_speakers import assign_speakers_to_words, to_runs

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

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
    ap.add_argument("--diarize-model", default="pyannote/speaker-diarization")
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
        write_jsonl(out_dir / f"{apath.stem}_words.jsonl", words)
        print(f"[asr] words: {len(words)}")

        # 2) Diarize
        spans_path = out_dir / "spans.jsonl"
        diarize_one(apath, spans_path, args.diarize_model, token, args.min_dur, args.bridge_gap)
        spans = [json.loads(line) for line in spans_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        # 3) Merge
        labeled = assign_speakers_to_words(words, spans, smooth_min_sec=0.6)
        write_jsonl(out_dir / f"{apath.stem}_merged.jsonl", labeled)
        runs = to_runs(labeled)
        with (out_dir / f"{apath.stem}_transcription.txt").open("w", encoding="utf-8") as f:
            for r in runs:
                f.write(f"[{r['speaker']}] {r['text'].strip()}\n")

        print(f"[done] {apath.name} in {time.time()-t0:.1f}s → {out_dir}")

if __name__ == "__main__":
    main()
