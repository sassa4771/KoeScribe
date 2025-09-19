# merge_segments.py
import json, argparse
from pathlib import Path
from typing import List, Dict, Any

def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows=[]
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def save_jsonl(p: Path, rows: List[Dict[str, Any]]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """[a0,a1] と [b0,b1] の重なり長さ（秒）"""
    return max(0.0, min(a1, b1) - max(a0, b0))

def assign_speaker_to_segments(
    segments: List[Dict[str, Any]],
    spans: List[Dict[str, Any]],
    min_overlap_ratio: float = 0.5,
    prefer_containment: bool = True,
) -> List[Dict[str, Any]]:
    """
    ルール:
      1) セグメントが完全に(収まって)含まれる span があればその speaker
      2) 無ければ、重なりが最大の span を採用（重なり/セグメント長 >= min_overlap_ratio）
      3) それでも無ければ、直前の speaker を継承。初回は "UNK"
    """
    spans = sorted(spans, key=lambda r: (r["start"], r["end"]))
    out=[]
    last_speaker = "UNK"

    for s in segments:
        s0 = float(s["start"]); s1 = float(s["end"])
        seg_len = max(1e-6, s1 - s0)

        chosen = None

        if prefer_containment:
            # 完全包含を優先
            for sp in spans:
                if sp["start"] <= s0 and s1 <= sp["end"]:
                    chosen = sp
                    break

        if chosen is None:
            # 最大重なりを探索
            best_ov = 0.0; best = None
            for sp in spans:
                ov = overlap(s0, s1, float(sp["start"]), float(sp["end"]))
                if ov > best_ov:
                    best_ov = ov; best = sp
            if best is not None and (best_ov / seg_len) >= min_overlap_ratio:
                chosen = best

        speaker = chosen["speaker"] if chosen else last_speaker
        out.append({**s, "speaker": speaker})
        last_speaker = speaker

    return out

def write_txt_grouped(p_out: Path, labeled_segments: List[Dict[str, Any]]):
    """連続する同一話者ごとに text を結合して .txt を出力"""
    if not labeled_segments:
        p_out.write_text("", encoding="utf-8"); return
    lines=[]
    cur_spk = labeled_segments[0].get("speaker", "UNK")
    cur_text = labeled_segments[0].get("text","")
    for seg in labeled_segments[1:]:
        spk = seg.get("speaker","UNK")
        if spk == cur_spk:
            cur_text += seg.get("text","")
        else:
            lines.append(f"[{cur_spk}] {cur_text.strip()}")
            cur_spk = spk
            cur_text = seg.get("text","")
    lines.append(f"[{cur_spk}] {cur_text.strip()}")
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Assign speaker labels to segments.jsonl using spans.jsonl")
    ap.add_argument("--segments", required=True, help="*_segments.jsonl (ASR 出力)")
    ap.add_argument("--spans",    required=True, help="spans.jsonl (pyannote 出力)")
    ap.add_argument("--out",      required=True, help="出力: segments に speaker を付けた JSONL")
    ap.add_argument("--txt",      help="話者ラベル付き transcription.txt を出力（任意）")
    ap.add_argument("--min-overlap-ratio", type=float, default=0.5,
                    help="重なり採用の最低割合（セグメント長に対する比率, 既定=0.5）")
    ap.add_argument("--no-containment-priority", action="store_true",
                    help="完全包含の優先を無効化（最大重なりのみで決定）")
    args = ap.parse_args()

    segs  = load_jsonl(Path(args.segments))
    spans = load_jsonl(Path(args.spans))

    labeled = assign_speaker_to_segments(
        segs, spans,
        min_overlap_ratio=args.min_overlap_ratio,
        prefer_containment=(not args.no_containment_priority),
    )
    save_jsonl(Path(args.out), labeled)

    if args.txt:
        write_txt_grouped(Path(args.txt), labeled)

    print(f"[merge-segments] wrote: {args.out}" + (f" & {args.txt}" if args.txt else ""))

if __name__ == "__main__":
    main()
