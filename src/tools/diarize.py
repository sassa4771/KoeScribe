# src/tools/diarize.py
import os, json, argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_csv(path: Path, rows: List[Dict[str, Any]]):
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

def check_token_authentication(model_id: str, token: Optional[str]) -> Tuple[bool, str]:
    """
    トークンの認証を確認する（モデルはダウンロードしない）
    
    Args:
        model_id: モデルID（例: 'pyannote/speaker-diarization'）
        token: Hugging Faceトークン
    
    Returns:
        (成功フラグ, メッセージ) のタプル
    """
    if not token:
        return False, "[ERROR] トークンが設定されていません。.envファイルにHUGGINGFACE_TOKENを設定してください。"
    
    try:
        from huggingface_hub import HfApi, login
        from huggingface_hub.utils import HfHubHTTPError
        
        # Hugging Face APIを使用してトークンを検証
        api = HfApi(token=token)
        
        # トークンでログインを試みる
        try:
            user_info = api.whoami()
            username = user_info.get("name", "不明")
            print(f"[OK] トークン認証成功: ユーザー '{username}'")
        except Exception as e:
            return False, f"[ERROR] トークン認証に失敗しました: {str(e)}\nトークンが無効または期限切れの可能性があります。"
        
        # モデルへのアクセス権限を確認
        try:
            model_info = api.model_info(model_id, token=token)
            print(f"[OK] モデル '{model_id}' へのアクセス権限を確認しました")
            return True, f"[OK] 認証成功！\n  ユーザー: {username}\n  モデル: {model_id}\n  アクセス: 許可されています"
        except HfHubHTTPError as e:
            if e.status_code == 403:
                return False, f"[ERROR] モデル '{model_id}' へのアクセスが拒否されました。\n  利用規約に同意しているか確認してください: https://huggingface.co/{model_id}"
            elif e.status_code == 404:
                return False, f"[ERROR] モデル '{model_id}' が見つかりません。\n  モデルIDが正しいか確認してください。"
            else:
                return False, f"[ERROR] モデルアクセス確認中にエラーが発生しました: {str(e)}"
        except Exception as e:
            return False, f"[ERROR] モデルアクセス確認中にエラーが発生しました: {str(e)}"
            
    except ImportError:
        # huggingface_hubがインストールされていない場合、pyannoteで直接確認を試みる
        try:
            from pyannote.audio import Pipeline
            
            # 実際にはモデルをダウンロードせず、認証だけを確認する
            # ただし、これはモデルのメタデータを取得するだけなので軽量
            try:
                if token:
                    # 新しいAPI（tokenパラメータ）のみを使用
                    Pipeline.from_pretrained(model_id, token=token)
                else:
                    Pipeline.from_pretrained(model_id)
                
                return True, f"[OK] 認証成功！\n  モデル: {model_id}\n  アクセス: 許可されています"
            except Exception as e:
                error_msg = str(e).lower()
                if "authentication" in error_msg or "token" in error_msg or "401" in error_msg or "403" in error_msg:
                    return False, f"[ERROR] 認証に失敗しました: {str(e)}\n  トークンが無効または、モデルの利用規約に同意していない可能性があります。"
                else:
                    return False, f"[ERROR] エラーが発生しました: {str(e)}"
        except Exception as e:
            return False, f"[ERROR] 認証確認中にエラーが発生しました: {str(e)}"

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

def diarize_one(audio_path: Path, out_path: Path, model_id: str, token: Optional[str],
                min_dur: float, bridge_gap: float):
    """
    話者分離を実行
    
    Args:
        token: Hugging Faceトークン。Noneの場合はローカルキャッシュから読み込みを試みる
              初回ダウンロード時のみ必要
    """
    from pyannote.audio import Pipeline
    
    # トークンがなくても、ローカルキャッシュがあれば使用可能
    # 新しいAPI（tokenパラメータ）のみを使用（use_auth_tokenは非推奨）
    try:
        if token:
            pipeline = Pipeline.from_pretrained(model_id, token=token)
        else:
            pipeline = Pipeline.from_pretrained(model_id)
    except Exception as e:
        error_msg = str(e)
        if not token:
            raise Exception(
                f"モデルの読み込みに失敗しました。初回ダウンロード時はHUGGINGFACE_TOKENが必要です。\n"
                f"エラー: {error_msg}\n"
                f"解決方法: .envファイルにHUGGINGFACE_TOKENを設定してください。\n"
                f"詳細: {model_id} モデルをダウンロードするには、Hugging Faceのアクセストークンが必要です。"
            )
        else:
            # より詳細なエラーメッセージ
            if "401" in error_msg or "authentication" in error_msg.lower():
                raise Exception(
                    f"認証に失敗しました: {error_msg}\n"
                    f"トークンが無効または期限切れの可能性があります。\n"
                    f"解決方法:\n"
                    f"  1. Hugging Faceのトークンが有効か確認してください\n"
                    f"  2. トークンを再生成してください: https://huggingface.co/settings/tokens\n"
                    f"  3. .envファイルのHUGGINGFACE_TOKENを更新してください"
                )
            elif "403" in error_msg or "access" in error_msg.lower():
                raise Exception(
                    f"アクセスが拒否されました: {error_msg}\n"
                    f"{model_id} モデルの利用規約に同意していない可能性があります。\n"
                    f"解決方法:\n"
                    f"  1. https://huggingface.co/{model_id} にアクセス\n"
                    f"  2. 利用規約に同意してください\n"
                    f"  3. 再度実行してください"
                )
            else:
                raise Exception(
                    f"モデルの読み込みに失敗しました: {error_msg}\n"
                    f"解決方法:\n"
                    f"  1. インターネット接続を確認してください\n"
                    f"  2. Hugging Faceのトークンが有効か確認してください\n"
                    f"  3. {model_id} モデルの利用規約に同意しているか確認してください"
                )
    
    # torchcodec が Windows で動作しないため、soundfile で事前ロードして渡す
    import soundfile as sf
    import torch
    waveform, sample_rate = sf.read(str(audio_path), always_2d=True)
    waveform_tensor = torch.tensor(waveform.T, dtype=torch.float32)
    diar = pipeline({"waveform": waveform_tensor, "sample_rate": sample_rate})
    # 新しい pyannote は DiarizeOutput を返す（speaker_diarization 属性に Annotation が入る）
    annotation = getattr(diar, 'speaker_diarization', None) or getattr(diar, 'annotation', diar)
    rows = [{"start": float(turn.start), "end": float(turn.end), "speaker": str(spk)}
            for turn, _, spk in annotation.itertracks(yield_label=True)]
    rows = postprocess_segments(rows, min_dur=min_dur, bridge_gap=bridge_gap)
    save_csv(out_path, rows)
    return out_path

def main():
    load_dotenv()  # .env を読み込む（HUGGINGFACE_TOKEN など）
    ap = argparse.ArgumentParser(description="Speaker diarization (pyannote) → spans.csv")
    ap.add_argument("--config", type=str, default="config.json", help="設定ファイル（root/files/out など）")
    ap.add_argument("--root", type=str, help="config の root_dir を上書き")
    ap.add_argument("--out", type=str, help="config の output_dir を上書き")
    ap.add_argument("--files", nargs="*", help="config の audio_files を上書き")
    ap.add_argument("--model", default=None, help="pyannote モデルID（既定は config か 'pyannote/speaker-diarization'）")
    ap.add_argument("--min-dur", type=float, default=None, help="区間の最小長(秒)（config を上書き）")
    ap.add_argument("--bridge-gap", type=float, default=None, help="短い隙間の連結しきい値(秒)（config を上書き）")
    ap.add_argument("--check-token", action="store_true", help="トークンの認証のみを確認（モデルはダウンロードしない）")
    args = ap.parse_args()
    
    # トークン認証確認モード
    if args.check_token:
        token = os.getenv("HUGGINGFACE_TOKEN")
        model_id = args.model or "pyannote/speaker-diarization-3.1"
        
        print("[info] トークン認証を確認中...")
        print(f"   モデル: {model_id}\n")
        
        success, message = check_token_authentication(model_id, token)
        print(f"\n{message}")
        
        if success:
            print("\n[OK] トークン認証は正常です。話者分離機能を使用できます。")
            return 0
        else:
            print("\n[ERROR] トークン認証に失敗しました。上記の解決方法を確認してください。")
            return 1

    # config 読み込み
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"設定ファイルが見つかりません: {cfg_path}")
    cfg = load_config(cfg_path)

    # 設定マージ（CLI引数があれば上書き）
    root_dir = Path(args.root or cfg.get("root_dir", "."))
    out_dir  = Path(args.out  or cfg.get("output_dir", "./outputs"))
    files    = args.files or cfg.get("audio_files", [])

    diar_model  = args.model or cfg.get("diarize_model", "pyannote/speaker-diarization-3.1")
    min_dur     = (args.min_dur if args.min_dur is not None else float(cfg.get("diarize_min_dur", 0.8)))
    bridge_gap  = (args.bridge_gap if args.bridge_gap is not None else float(cfg.get("diarize_bridge_gap", 0.3)))

    if not files:
        raise SystemExit("audio_files が空です。config.json に列挙するか、--files で指定してください。")

    token = os.getenv("HUGGINGFACE_TOKEN")
    # トークンがなくても、ローカルキャッシュがあれば動作可能
    # ただし、初回ダウンロード時はトークンが必要
    if not token:
        print("[warning] HUGGINGFACE_TOKEN が未設定です。")
        print("  ローカルキャッシュからモデルを読み込みます。")
        print("  初回ダウンロード時はトークンが必要です。")

    # 入力存在チェック
    inputs = [root_dir / f for f in files]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise SystemExit("音声ファイルが見つかりません:\n- " + "\n- ".join(missing))

    for p in tqdm(inputs, desc="🗣️ Diarizing", unit="file"):
        spans_path = out_dir / f"output_{p.stem}" / "spans.csv"
        wrote = diarize_one(p, spans_path, diar_model, token, min_dur, bridge_gap)
        print(f"[diarize] wrote: {wrote}")

if __name__ == "__main__":
    main()
