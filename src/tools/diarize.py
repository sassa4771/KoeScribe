# src/tools/diarize.py
import os, json, argparse
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def postprocess_segments(rows, min_dur=0.8, bridge_gap=0.3):
    """短区間の吸収＆短隙間ブリッジで安定化。"""
    if not rows:
        return rows
    rows = sorted(rows, key=lambda r: (r["start"], r["end"]))
    merged = []
    cur = rows[0].copy()
    for r in rows[1:]:
        # 同一話者で隙間が小さいなら連結
        if (r["speaker"] == cur["speaker"]) and (r["start"] - cur["end"] <= bridge_gap):
            cur["end"] = max(cur["end"], r["end"])
        else:
            # 短すぎる区間は前後に吸収を試みる（簡易）
            if (cur["end"] - cur["start"] < min_dur) and merged:
                prev = merged[-1]
                if (prev["speaker"] == cur["speaker"]) and (cur["start"] - prev["end"] <= bridge_gap):
                    prev["end"] = max(prev["end"], cur["end"])
                else:
                    merged.append(cur)
            else:
                merged.append(cur)
            cur = r.copy()
    merged.append(cur)
    return merged

def diarize_one(audio_path: Path, out_path: Path, model_id: str, token: str,
                min_dur: float, bridge_gap: float):
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)
    diar = pipeline(str(audio_path))  # pyannote.core.Annotation
    rows = [{"start": float(turn.start), "end": float(turn.end), "speaker": str(spk)}
            for turn, _, spk in diar.itertracks(yield_label=True)]
    rows = postprocess_segments(rows, min_dur=min_dur, bridge_gap=bridge_gap)
    save_jsonl(out_path, rows)
    return out_path

def main():
    load_dotenv()  # .env を読み込む（HUGGINGFACE_TOKEN など）
    ap = argparse.ArgumentParser(description="Speaker diarization (pyannote) → spans.jsonl")
    ap.add_argument("--config", type=str, default="config.json", help="設定ファイル（root/files/out など）")
    ap.add_argument("--root", type=str, help="config の root_dir を上書き")
    ap.add_argument("--out", type=str, help="config の output_dir を上書き")
    ap.add_argument("--files", nargs="*", help="config の audio_files を上書き")
    ap.add_argument("--model", default=None, help="pyannote モデルID（既定は config か 'pyannote/speaker-diarization'）")
    ap.add_argument("--min-dur", type=float, default=None, help="区間の最小長(秒)（config を上書き）")
    ap.add_argument("--bridge-gap", type=float, default=None, help="短い隙間の連結しきい値(秒)（config を上書き）")
    args = ap.parse_args()

    # config 読み込み
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"設定ファイルが見つかりません: {cfg_path}")
    cfg = load_config(cfg_path)

    # 設定マージ（CLI引数があれば上書き）
    root_dir = Path(args.root or cfg.get("root_dir", "."))
    out_dir  = Path(args.out  or cfg.get("output_dir", "./outputs"))
    files    = args.files or cfg.get("audio_files", [])

    diar_model  = args.model or cfg.get("diarize_model", "pyannote/speaker-diarization")
    min_dur     = (args.min_dur if args.min_dur is not None else float(cfg.get("diarize_min_dur", 0.8)))
    bridge_gap  = (args.bridge_gap if args.bridge_gap is not None else float(cfg.get("diarize_bridge_gap", 0.3)))

    if not files:
        raise SystemExit("audio_files が空です。config.json に列挙するか、--files で指定してください。")

    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit("HUGGINGFACE_TOKEN が未設定です。プロジェクト直下の .env に記載してください。")

    # 入力存在チェック
    inputs = [root_dir / f for f in files]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise SystemExit("音声ファイルが見つかりません:\n- " + "\n- ".join(missing))

    for p in tqdm(inputs, desc="🗣️ Diarizing", unit="file"):
        spans_path = out_dir / f"output_{p.stem}" / "spans.jsonl"
        wrote = diarize_one(p, spans_path, diar_model, token, min_dur, bridge_gap)
        print(f"[diarize] wrote: {wrote}")

if __name__ == "__main__":
    main()
