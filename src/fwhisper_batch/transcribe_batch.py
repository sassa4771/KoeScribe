# src/fwhisper_batch/transcribe_batch.py
import os
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from faster_whisper import WhisperModel
import ctranslate2
from tqdm import tqdm


def detect_device() -> str:
    try:
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def resolve_device(device_cfg: str) -> str:
    if device_cfg and device_cfg.lower() in {"cuda", "cpu"}:
        return device_cfg.lower()
    env = os.getenv("FWHISPER_DEVICE", "").lower()
    if env in {"cuda", "cpu"}:
        return env
    return detect_device()


def resolve_compute_type(device: str, compute_cfg: str) -> str:
    if compute_cfg and compute_cfg.lower() != "auto":
        return compute_cfg
    env = os.getenv("FWHISPER_COMPUTE", "").lower()
    if env:
        return env
    return "int8_float16" if device == "cuda" else "int8"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def transcribe_one(
    model: WhisperModel,
    audio_path: Path,
    out_dir: Path,
    language: str,
    beam_size: int,
    use_vad: bool,
    min_silence_ms: int,
    word_timestamps: bool,
    show_progress: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    segments, info = model.transcribe(
        str(audio_path),
        language=language if language else None,
        beam_size=beam_size,
        vad_filter=use_vad,
        vad_parameters={"min_silence_duration_ms": min_silence_ms},
        word_timestamps=word_timestamps,
    )

    # ✅ セグメント逐次処理＋進捗バー
    rows: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    dur = float(getattr(info, "duration", 0.0) or 0.0)
    last_end = 0.0

    pbar = None
    if show_progress and dur > 0:
        pbar = tqdm(
            total=dur,
            unit="s",
            desc=audio_path.name,
            leave=False,
            bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s",
        )

    for s in segments:
        row: Dict[str, Any] = {"start": s.start, "end": s.end, "text": s.text}
        if word_timestamps and getattr(s, "words", None):
            row["words"] = [{"start": w.start, "end": w.end, "word": w.word} for w in s.words]
        rows.append(row)
        text_parts.append(s.text)

        if pbar is not None:
            inc = max(0.0, float(s.end) - last_end)
            if inc > 0:
                pbar.update(inc)
                last_end = float(s.end)

    if pbar is not None:
        pbar.close()

    text = "".join(text_parts)
    end = time.time()
    proc_time = end - start

    # 保存
    (out_dir / f"{audio_path.stem}_transcription.txt").write_text(text, encoding="utf-8")
    stats = [
        f"処理時間: {proc_time:.2f} 秒",
        f"推定言語: {info.language}",
        f"音声長: {info.duration:.2f} 秒",
    ]
    (out_dir / f"{audio_path.stem}_processing_time.txt").write_text("\n".join(stats) + "\n", encoding="utf-8")
    write_jsonl(out_dir / f"{audio_path.stem}_segments.jsonl", rows)

    return {
        "text": text,
        "processing_time": proc_time,
        "language": info.language,
        "duration": info.duration,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch transcription with Faster-Whisper (uv)")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--root", type=str, help="Override root_dir")
    parser.add_argument("--out", type=str, help="Override output_dir")
    parser.add_argument("--files", nargs="*", help="Override audio files list")
    parser.add_argument("--word-timestamps", action="store_true", help="Include per-word timestamps in JSONL")
    parser.add_argument("--no-progress", action="store_true", help="Disable per-file progress bar")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    root_dir = Path(args.root or cfg.get("root_dir", "."))
    out_dir = Path(args.out or cfg.get("output_dir", "./outputs"))
    files = args.files or cfg.get("audio_files", [])

    model_size = cfg.get("model_size", "large-v3")
    device_cfg = cfg.get("device", "auto")
    compute_cfg = cfg.get("compute_type", "auto")
    beam_size = int(cfg.get("beam_size", 5))
    use_vad = bool(cfg.get("use_vad", True))
    min_silence = int(cfg.get("min_silence_ms", 500))
    language = cfg.get("language", "ja")

    device = resolve_device(device_cfg)
    compute_type = resolve_compute_type(device, compute_cfg)

    print(f"[fwhisper] model={model_size} device={device} compute_type={compute_type}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    inputs = [root_dir / f for f in files]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise SystemExit("Input files not found:\n- " + "\n- ".join(missing))

    for idx, p in enumerate(tqdm(inputs, desc="🎧 Transcribing", unit="file"), start=1):
        target_out = out_dir / f"output_{p.stem}"
        result = transcribe_one(
            model,
            p,
            target_out,
            language,
            beam_size,
            use_vad,
            min_silence,
            args.word_timestamps,
            show_progress=(not args.no_progress),
        )
        preview = result["text"][:500]
        print(f"\n=== {idx}/{len(inputs)} Done: {p.name} ===")
        print(preview + ("..." if len(result["text"]) > len(preview) else ""))


if __name__ == "__main__":
    main()
