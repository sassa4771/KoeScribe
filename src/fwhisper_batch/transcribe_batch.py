# src/fwhisper_batch/transcribe_batch.py
import os
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from faster_whisper import WhisperModel
import ctranslate2
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd

from fwhisper_batch.video_converter import VideoConverter


def detect_device() -> str:
    try:
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def resolve_device(device_cfg: str) -> str:
    if device_cfg and device_cfg.lower() in {"cuda", "cpu", "auto"}:
        if device_cfg.lower() == "auto":
            return detect_device()
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


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding='utf-8-sig')


def is_diarization_enabled(cfg: Dict[str, Any]) -> bool:
    """Check if diarization is enabled based on config"""
    # トークンがなくても、ローカルキャッシュがあれば動作可能
    # ただし、初回ダウンロード時はトークンが必要
    return bool(cfg.get("diarize_model"))


def diarize_and_merge(audio_path: Path, out_dir: Path, cfg: Dict[str, Any], words: List[Dict[str, Any]], segments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Perform diarization and merge with ASR results"""
    try:
        from tools.diarize import diarize_one
        from tools.merge_speakers import assign_speakers_to_words
        
        token = os.getenv("HUGGINGFACE_TOKEN")
        # トークンがなくても、ローカルキャッシュがあれば動作可能
        # ただし、初回ダウンロード時はトークンが必要
        
        spans_path = out_dir / "spans.csv"
        diarize_model = cfg.get("diarize_model", "pyannote/speaker-diarization")
        min_dur = float(cfg.get("diarize_min_dur", 0.8))
        bridge_gap = float(cfg.get("diarize_bridge_gap", 0.3))
        
        try:
            diarize_one(audio_path, spans_path, diarize_model, token, min_dur, bridge_gap)
        except Exception as e:
            if not token:
                print(f"[warning] HUGGINGFACE_TOKEN not set, and model not in local cache.")
                print(f"  初回ダウンロード時はトークンが必要です。エラー: {str(e)}")
            else:
                print(f"[warning] Diarization failed: {str(e)}")
            return None
        
        spans = []
        if spans_path.exists():
            df = pd.read_csv(spans_path, encoding='utf-8-sig')
            spans = df.to_dict('records')
        
        labeled_words = assign_speakers_to_words(words, spans, smooth_min_sec=0.6)
        write_csv(out_dir / f"{audio_path.stem}_words_with_speakers.csv", labeled_words)
        
        labeled_segments = assign_speakers_to_words(segments, spans, smooth_min_sec=0.6)
        write_csv(out_dir / f"{audio_path.stem}_segments_with_speakers.csv", labeled_segments)
        
        return {
            "words_with_speakers": len(labeled_words),
            "segments_with_speakers": len(labeled_segments)
        }
        
    except ImportError as e:
        print(f"[warning] Diarization dependencies not available: {e}")
        return None
    except Exception as e:
        print(f"[warning] Diarization failed for {audio_path.name}: {e}")
        return None


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
    enable_diarization: bool = False,
    diarization_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    1ファイルを文字起こしして出力一式を保存する。
    - CSV: <stem>_segments.csv
    - CSV: <stem>_words.csv（word_timestamps=True の時だけ）
    - TXT: <stem>_processing_time.txt
    - CSV: <stem>_words_with_speakers.csv（enable_diarization=True の時だけ）
    - CSV: <stem>_segments_with_speakers.csv（enable_diarization=True の時だけ）
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    segments, info = model.transcribe(
        str(audio_path),
        language=language if language else None,
        beam_size=beam_size,
        vad_filter=use_vad,
        vad_parameters={"min_silence_duration_ms": min_silence_ms} if use_vad else None,
        word_timestamps=word_timestamps,
    )

    # 逐次処理＋進捗
    rows_segments: List[Dict[str, Any]] = []
    rows_words: List[Dict[str, Any]] = []
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
        # セグメント保存
        seg_obj: Dict[str, Any] = {"start": s.start, "end": s.end, "text": s.text}
        text_parts.append(s.text)
        # 単語があれば words 側に格納（セグメントにも入れないで軽量化）
        if word_timestamps and getattr(s, "words", None):
            for w in s.words:
                rows_words.append({"start": w.start, "end": w.end, "word": w.word})
        rows_segments.append(seg_obj)

        # 進捗更新
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
    write_csv(out_dir / f"{audio_path.stem}_segments.csv", rows_segments)
    if word_timestamps and rows_words:
        write_csv(out_dir / f"{audio_path.stem}_words.csv", rows_words)

    stats = [
        f"処理時間: {proc_time:.2f} 秒",
        f"推定言語: {getattr(info, 'language', '')}",
        f"音声長: {getattr(info, 'duration', 0.0):.2f} 秒",
        f"単語タイムスタンプ: {'あり' if (word_timestamps and rows_words) else 'なし'}",
        f"VAD: {'ON' if use_vad else 'OFF'} (min_silence_ms={min_silence_ms if use_vad else 'N/A'})",
    ]
    (out_dir / f"{audio_path.stem}_processing_time.txt").write_text("\n".join(stats) + "\n", encoding="utf-8")

    result = {
        "text": text,
        "processing_time": proc_time,
        "language": getattr(info, "language", None),
        "duration": getattr(info, "duration", None),
        "words_count": len(rows_words),
    }
    
    if enable_diarization and diarization_config and rows_words:
        diarization_result = diarize_and_merge(audio_path, out_dir, diarization_config, rows_words, rows_segments)
        if diarization_result:
            result["diarization"] = diarization_result
    
    return result


def transcribe_one_with_callback(
    model: WhisperModel,
    audio_path: Path,
    out_dir: Path,
    language: str,
    beam_size: int,
    use_vad: bool,
    min_silence_ms: int,
    word_timestamps: bool,
    show_progress: bool = True,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Transcribe one file with progress callback support for GUI integration
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    segments, info = model.transcribe(
        str(audio_path),
        language=language if language else None,
        beam_size=beam_size,
        vad_filter=use_vad,
        vad_parameters={"min_silence_duration_ms": min_silence_ms} if use_vad else None,
        word_timestamps=word_timestamps,
    )

    rows_segments: List[Dict[str, Any]] = []
    rows_words: List[Dict[str, Any]] = []
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
        seg_obj: Dict[str, Any] = {"start": s.start, "end": s.end, "text": s.text}
        text_parts.append(s.text)
        
        if word_timestamps and getattr(s, "words", None):
            for w in s.words:
                rows_words.append({"start": w.start, "end": w.end, "word": w.word})
        rows_segments.append(seg_obj)

        if dur > 0:
            progress_percent = min(100.0, (float(s.end) / dur) * 100.0)
            if progress_callback:
                progress_callback(progress_percent)
                
        if pbar is not None:
            inc = max(0.0, float(s.end) - last_end)
            if inc > 0:
                pbar.update(inc)
                last_end = float(s.end)

    if pbar is not None:
        pbar.close()
        
    if progress_callback:
        progress_callback(100.0)

    text = "".join(text_parts)
    end = time.time()
    proc_time = end - start

    write_csv(out_dir / f"{audio_path.stem}_segments.csv", rows_segments)
    if word_timestamps and rows_words:
        write_csv(out_dir / f"{audio_path.stem}_words.csv", rows_words)

    stats = [
        f"処理時間: {proc_time:.2f} 秒",
        f"推定言語: {getattr(info, 'language', '')}",
        f"音声長: {getattr(info, 'duration', 0.0):.2f} 秒",
        f"単語タイムスタンプ: {'あり' if (word_timestamps and rows_words) else 'なし'}",
        f"VAD: {'ON' if use_vad else 'OFF'} (min_silence_ms={min_silence_ms if use_vad else 'N/A'})",
    ]
    (out_dir / f"{audio_path.stem}_processing_time.txt").write_text("\n".join(stats) + "\n", encoding="utf-8")

    return {
        "text": text,
        "processing_time": proc_time,
        "language": getattr(info, "language", None),
        "duration": getattr(info, "duration", None),
        "words_count": len(rows_words),
    }


def main():
    load_dotenv()  # Load .env file for HUGGINGFACE_TOKEN
    parser = argparse.ArgumentParser(description="Batch transcription with Faster-Whisper (uv)")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--root", type=str, help="Override root_dir")
    parser.add_argument("--out", type=str, help="Override output_dir")
    parser.add_argument("--files", nargs="*", help="Override audio files list")
    parser.add_argument("--word-timestamps", action="store_true", help="Export per-word timestamps (also writes *_words.csv)")
    parser.add_argument("--no-progress", action="store_true", help="Disable per-file progress bar")
    parser.add_argument("--disable-diarization", action="store_true", help="Disable speaker diarization even if configured")
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

    enable_diarization = (not args.disable_diarization) and is_diarization_enabled(cfg)
    if enable_diarization:
        print("[info] Speaker diarization enabled")
    else:
        print("[info] Speaker diarization disabled")

    print(f"[koescribe] model={model_size} device={device} compute_type={compute_type}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    inputs = [root_dir / f for f in files]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise SystemExit("Input files not found:\n- " + "\n- ".join(missing))

    if not VideoConverter.is_ffmpeg_available():
        print("[warning] ffmpeg not found. Video conversion will not be available.")
        print("[warning] Only WAV files will be processed.")
    
    temp_audio_dir = out_dir / "converted_audio"

    for idx, p in enumerate(tqdm(inputs, desc="🎧 Transcribing", unit="file"), start=1):
        target_out = out_dir / f"output_{p.stem}"
        
        audio_file = p
        was_converted = False
        
        try:
            if p.suffix.lower() != '.wav':
                print(f"\n[info] Converting {p.name} to WAV...")
                audio_file, was_converted = VideoConverter.prepare_file_for_transcription(p, temp_audio_dir)
                if was_converted:
                    print(f"[info] Converted to: {audio_file.name}")
        except Exception as e:
            print(f"\n[error] Failed to convert {p.name}: {e}")
            print(f"[error] Skipping {p.name}")
            continue
        
        use_word_timestamps = args.word_timestamps or enable_diarization
        
        result = transcribe_one(
            model,
            audio_file,
            target_out,
            language,
            beam_size,
            use_vad,
            min_silence,
            use_word_timestamps,
            show_progress=(not args.no_progress),
            enable_diarization=enable_diarization,
            diarization_config=cfg if enable_diarization else None,
        )
        preview = result["text"][:500]
        print(f"\n=== {idx}/{len(inputs)} Done: {p.name} ===")
        print(preview + ("..." if len(result["text"]) > len(preview) else ""))
        if use_word_timestamps:
            print(f"words.csv: {result['words_count']} words")
        if enable_diarization and result.get("diarization"):
            diar_info = result["diarization"]
            print(f"diarization: {diar_info['words_with_speakers']} words, {diar_info['segments_with_speakers']} segments with speakers")

if __name__ == "__main__":
    main()
