# merge_segments.py
import json, argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

def load_csv(p: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(p, encoding='utf-8-sig')
    return df.to_dict('records')

def save_csv(p: Path, rows: List[Dict[str, Any]]):
    """
    CSVファイルを書き込む。権限エラーなどの問題を適切に処理する。
    """
    try:
        # ディレクトリを作成
        p.parent.mkdir(parents=True, exist_ok=True)
        
        if rows:
            df = pd.DataFrame(rows)
            
            # 既存のファイルが読み取り専用の場合、属性を変更してから削除を試みる
            if p.exists():
                try:
                    # Windowsで読み取り専用属性を解除
                    import os
                    if os.name == 'nt':  # Windows
                        import stat
                        current_attrs = p.stat().st_file_attributes
                        if current_attrs & stat.FILE_ATTRIBUTE_READONLY:
                            p.chmod(stat.S_IWRITE)
                except Exception:
                    pass  # 属性変更に失敗しても続行
            
            # 一時ファイルに書き込んでからリネーム（アトミック書き込み）
            temp_path = p.with_suffix('.tmp')
            try:
                df.to_csv(temp_path, index=False, encoding='utf-8-sig')
                # 既存ファイルがあれば削除
                if p.exists():
                    p.unlink()
                # 一時ファイルをリネーム
                temp_path.rename(p)
            except PermissionError as e:
                # 一時ファイルをクリーンアップ
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass
                raise PermissionError(
                    f"ファイル '{p}' への書き込み権限がありません。\n"
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
            f"CSVファイル '{p}' の書き込みに失敗しました: {str(e)}\n"
            f"ファイルパス: {p}\n"
            f"ディレクトリの書き込み権限を確認してください。"
        )

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

def assign_speakers_to_words(words: List[Dict[str, Any]], spans: List[Dict[str, Any]], smooth_min_sec: float = 0.6) -> List[Dict[str, Any]]:
    """単語レベルで話者を割り当て（pipeline_all.py用）"""
    return assign_speaker_to_segments(words, spans, min_overlap_ratio=0.3, prefer_containment=True)

def to_runs(labeled_words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """連続する同一話者の単語をまとめてrunに変換（pipeline_all.py用）"""
    if not labeled_words:
        return []
    
    runs = []
    current_speaker = labeled_words[0].get("speaker", "UNK")
    current_text = labeled_words[0].get("word", "")
    current_start = labeled_words[0].get("start", 0.0)
    current_end = labeled_words[0].get("end", 0.0)
    
    for word in labeled_words[1:]:
        speaker = word.get("speaker", "UNK")
        if speaker == current_speaker:
            current_text += word.get("word", "")
            current_end = word.get("end", current_end)
        else:
            runs.append({
                "speaker": current_speaker,
                "text": current_text,
                "start": current_start,
                "end": current_end
            })
            current_speaker = speaker
            current_text = word.get("word", "")
            current_start = word.get("start", 0.0)
            current_end = word.get("end", 0.0)
    
    runs.append({
        "speaker": current_speaker,
        "text": current_text,
        "start": current_start,
        "end": current_end
    })
    
    return runs

def main():
    ap = argparse.ArgumentParser(description="Assign speaker labels to segments.csv using spans.csv")
    ap.add_argument("--segments", required=True, help="*_segments.csv (ASR 出力)")
    ap.add_argument("--spans",    required=True, help="spans.csv (pyannote 出力)")
    ap.add_argument("--out",      required=True, help="出力: segments に speaker を付けた CSV")
    ap.add_argument("--txt",      help="話者ラベル付き transcription.txt を出力（任意）")
    ap.add_argument("--min-overlap-ratio", type=float, default=0.5,
                    help="重なり採用の最低割合（セグメント長に対する比率, 既定=0.5）")
    ap.add_argument("--no-containment-priority", action="store_true",
                    help="完全包含の優先を無効化（最大重なりのみで決定）")
    args = ap.parse_args()

    segs  = load_csv(Path(args.segments))
    spans = load_csv(Path(args.spans))

    labeled = assign_speaker_to_segments(
        segs, spans,
        min_overlap_ratio=args.min_overlap_ratio,
        prefer_containment=(not args.no_containment_priority),
    )
    save_csv(Path(args.out), labeled)

    if args.txt:
        write_txt_grouped(Path(args.txt), labeled)

    print(f"[merge-segments] wrote: {args.out}" + (f" & {args.txt}" if args.txt else ""))

if __name__ == "__main__":
    main()
