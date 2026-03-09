import sys
import os
import json
import threading
import time
import queue
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QProgressBar, QFrame, QScrollArea,
    QGroupBox, QSpinBox, QLineEdit, QComboBox,
    QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QCheckBox
)
from PySide6.QtCore import QThread, Signal, QTimer, Qt, QSettings, QUrl
from PySide6.QtGui import QFont, QIcon, QDesktopServices

import pandas as pd
from dotenv import load_dotenv
from faster_whisper import WhisperModel

from fwhisper_batch.transcribe_batch import (
    transcribe_one, transcribe_one_with_callback, load_config, resolve_device, resolve_compute_type,
    detect_device, is_diarization_enabled, diarize_and_merge
)
from fwhisper_batch.video_converter import VideoConverter


def load_env_file():
    """Load .env file from executable directory or current directory"""
    if getattr(sys, 'frozen', False):
        exe_dir = Path(sys.executable).parent
        env_path = exe_dir / '.env'
        if env_path.exists():
            _load_env_with_encoding(env_path)
        else:
            _load_env_with_encoding()
    else:
        _load_env_with_encoding()


def _load_env_with_encoding(env_path=None):
    """複数のエンコーディングを試して.envファイルを読み込む"""
    encodings = ['utf-8', 'utf-8-sig', 'shift_jis', 'cp932']

    for encoding in encodings:
        try:
            if env_path:
                with open(env_path, 'r', encoding=encoding) as f:
                    load_dotenv(stream=f, override=True)
                return
            else:
                env_file = Path('.env')
                if env_file.exists():
                    with open(env_file, 'r', encoding=encoding) as f:
                        load_dotenv(stream=f, override=True)
                    return
                else:
                    load_dotenv()
                    return
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            if env_path is None:
                load_dotenv()
            return

    if env_path:
        print(f"[warning] Failed to load .env file with multiple encodings: {env_path}")
    else:
        print("[warning] Failed to load .env file with multiple encodings")


HELP_TOOLTIPS = {
    "beam_size": "ビームサーチのサイズ。大きいほど精度が向上しますが処理時間が増加します。\n推奨値: 1-10 (デフォルト: 5)",
    "model_size": "Whisperモデルのサイズ。large-v3が最も精度が高く、smallが最も高速です。\n選択肢: tiny, base, small, medium, large-v3",
    "compute_type": "計算精度。float16はGPU推奨、int8はメモリ使用量を削減します。\n選択肢: float16, int8_float16, int8",
    "min_silence_ms": "VAD（音声検出）で無音と判定する最小時間（ミリ秒）。\n短いほど細かく検出、長いほど安定します。",
    "max_speakers": "話者分離で検出する最大話者数。実際の話者数に近い値を設定してください。",
    "device": "処理に使用するデバイス。autoは自動選択、cudaはGPU、cpuはCPUを使用します。",
    "language": "音声の言語。jaは日本語、enは英語、autoは自動検出です。",
    "use_vad": "音声活動検出（VAD）を使用するかどうか。ONにすると無音部分を自動検出します。"
}


@dataclass
class SettingsPreset:
    name: str
    model_size: str = "large-v3"
    language: str = "ja"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 5
    use_vad: bool = True
    min_silence_ms: int = 500
    enable_diarization: bool = True
    diarize_model: str = "pyannote/speaker-diarization-3.1"
    diarize_min_dur: float = 0.8
    diarize_bridge_gap: float = 0.3
    max_speakers: int = 2
    output_dir: str = "./outputs"


@dataclass
class ProcessingJob:
    file_path: Path
    settings_preset: SettingsPreset
    status: str = "待機中"
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    job_id: Optional[int] = None


class ResultsDatabase:
    def __init__(self, db_path: Path = Path.home() / ".fwhisper_results.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processing_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                project_name TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT,
                processing_time REAL,
                estimated_language TEXT,
                audio_duration REAL,
                word_timestamps TEXT,
                vad_settings TEXT,
                output_directory TEXT,
                error_message TEXT
            )
        """)
        conn.commit()
        conn.close()

    def save_result(self, job: ProcessingJob) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        estimated_language = job.result.get('language', '') if job.result else ''
        audio_duration = job.result.get('duration', 0) if job.result else 0
        word_timestamps = 'あり' if job.result and job.result.get('words_count', 0) > 0 else 'なし'
        vad_settings = f"VAD: {'ON' if job.settings_preset.use_vad else 'OFF'}"
        if job.settings_preset.use_vad:
            vad_settings += f" (min_silence_ms={job.settings_preset.min_silence_ms})"

        cursor.execute("""
            INSERT INTO processing_results
            (file_path, project_name, start_time, end_time, status, processing_time, estimated_language, audio_duration, word_timestamps, vad_settings, output_directory, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(job.file_path),
            job.settings_preset.name,
            datetime.now(),
            datetime.now(),
            job.status,
            job.result.get('processing_time', 0) if job.result else 0,
            estimated_language,
            audio_duration,
            word_timestamps,
            vad_settings,
            job.settings_preset.output_dir,
            job.error
        ))
        job_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return job_id

    def get_recent_results(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM processing_results
            ORDER BY start_time DESC
            LIMIT ?
        """, (limit,))

        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return results


class TranscriptionWorker(QThread):
    progress_updated = Signal(str, float, str)
    phase_progress_updated = Signal(str, int, float, str)  # file_path, phase_index, progress, phase_name
    timer_updated = Signal(str, float)  # file_path, elapsed_seconds
    job_completed = Signal(str, dict)
    job_failed = Signal(str, str)
    queue_updated = Signal(int)
    diarization_skipped = Signal(str, str)  # file_path, reason

    def __init__(self):
        super().__init__()
        self.jobs = queue.Queue()
        self.current_job: Optional[ProcessingJob] = None
        self.model: Optional[WhisperModel] = None
        self.should_stop = False
        self.results_db = ResultsDatabase()

    def add_job(self, job: ProcessingJob):
        self.jobs.put(job)
        self.queue_updated.emit(self.jobs.qsize())

    def stop_processing(self):
        self.should_stop = True
        try:
            self.jobs.put_nowait(None)
        except:
            pass

    def run(self):
        self.should_stop = False
        while not self.should_stop:
            try:
                job = self.jobs.get(block=True, timeout=None)
                if job is None:
                    break
                if self.should_stop:
                    break
                self.current_job = job
                self.process_job(job)
                self.jobs.task_done()
                self.queue_updated.emit(self.jobs.qsize())
                self.current_job = None
            except Exception as e:
                print(f"Error in worker thread: {e}")
                import traceback
                traceback.print_exc()
                self.current_job = None
                continue

        self.current_job = None

    def process_job(self, job: ProcessingJob):
        try:
            import time
            job_start_time = time.time()

            audio_file = job.file_path
            was_converted = False
            temp_audio_dir = None

            if job.file_path.suffix.lower() != '.wav':
                self.phase_progress_updated.emit(str(job.file_path), 0, 0.0, "WAV変換開始")

                try:
                    temp_audio_dir = Path(job.settings_preset.output_dir) / "converted_audio"
                    audio_file, was_converted = VideoConverter.prepare_file_for_transcription(
                        job.file_path, temp_audio_dir
                    )
                    self.phase_progress_updated.emit(str(job.file_path), 0, 100.0, "WAV変換完了")
                except Exception as e:
                    raise Exception(f"動画変換エラー: {str(e)}")
            else:
                self.phase_progress_updated.emit(str(job.file_path), 0, 100.0, "WAV変換スキップ")

            self.phase_progress_updated.emit(str(job.file_path), 1, 0.0, "モデル読み込み中")

            if self.model is None:
                device = resolve_device(job.settings_preset.device)
                compute_type = resolve_compute_type(device, job.settings_preset.compute_type)
                self.model = WhisperModel(
                    job.settings_preset.model_size,
                    device=device,
                    compute_type=compute_type
                )

            output_dir = Path(job.settings_preset.output_dir) / f"output_{job.file_path.stem}"
            output_dir.mkdir(parents=True, exist_ok=True)

            self.phase_progress_updated.emit(str(job.file_path), 1, 0.0, "文字起こし開始")

            def transcription_progress_callback(progress: float):
                self.phase_progress_updated.emit(str(job.file_path), 1, progress, "文字起こし中")

            result = self._transcribe_with_progress(
                job, audio_file, output_dir, transcription_progress_callback
            )

            self.phase_progress_updated.emit(str(job.file_path), 1, 100.0, "文字起こし完了")

            if job.settings_preset.enable_diarization:
                self.phase_progress_updated.emit(str(job.file_path), 2, 0.0, "話者分離開始")

                diarization_result = self._perform_diarization(job, audio_file, output_dir, result)
                if diarization_result:
                    result["diarization"] = diarization_result
                    self.phase_progress_updated.emit(str(job.file_path), 2, 100.0, "話者分離完了")
                else:
                    self.phase_progress_updated.emit(str(job.file_path), 2, 100.0, "話者分離スキップ")
            else:
                self.phase_progress_updated.emit(str(job.file_path), 2, 100.0, "話者分離無効")

            self.phase_progress_updated.emit(str(job.file_path), 3, 0.0, "CSV変換開始")
            self._convert_to_csv(output_dir, job.file_path.stem, result)
            self.phase_progress_updated.emit(str(job.file_path), 3, 100.0, "CSV変換完了")

            job.result = result

            if job.settings_preset.enable_diarization and not result.get('diarization'):
                job.status = "完了 (話者分離スキップ)"
                self.progress_updated.emit(str(job.file_path), 100.0, "完了 (話者分離スキップ)")
            else:
                job.status = "完了"
                self.progress_updated.emit(str(job.file_path), 100.0, "完了")

            job.job_id = self.results_db.save_result(job)
            self.job_completed.emit(str(job.file_path), result)

        except Exception as e:
            job.error = str(e)
            job.status = "エラー"
            job.job_id = self.results_db.save_result(job)
            self.job_failed.emit(str(job.file_path), str(e))

    def _transcribe_with_progress(self, job: ProcessingJob, audio_file: Path, output_dir: Path, progress_callback):
        return transcribe_one_with_callback(
            self.model,
            audio_file,
            output_dir,
            job.settings_preset.language,
            job.settings_preset.beam_size,
            job.settings_preset.use_vad,
            job.settings_preset.min_silence_ms,
            word_timestamps=True,
            show_progress=False,
            progress_callback=progress_callback
        )

    def _perform_diarization(self, job: ProcessingJob, audio_file: Path, output_dir: Path, transcription_result: Dict[str, Any]):
        if not job.settings_preset.enable_diarization:
            return None

        try:
            import time
            import os

            token = os.getenv("HUGGINGFACE_TOKEN")
            has_token = bool(token)

            self.phase_progress_updated.emit(str(job.file_path), 2, 10.0, "話者分離モデル読み込み中")
            time.sleep(0.1)

            words_file = output_dir / f"{audio_file.stem}_words.csv"
            segments_file = output_dir / f"{audio_file.stem}_segments.csv"

            missing_files = []
            if not words_file.exists():
                missing_files.append(f"words.csv ({words_file})")
            if not segments_file.exists():
                missing_files.append(f"segments.csv ({segments_file})")

            if missing_files:
                error_msg = f"話者分離に必要なファイルが見つかりません:\n" + "\n".join(f"  - {f}" for f in missing_files)
                print(f"[warning] {error_msg}")
                return None

            if words_file.stat().st_size == 0:
                print(f"[warning] {words_file} が空です。")
                return None

            self.phase_progress_updated.emit(str(job.file_path), 2, 30.0, "音声データ解析中")

            words_df = pd.read_csv(words_file, encoding='utf-8-sig')
            words = words_df.to_dict('records')

            if not words:
                print(f"[warning] {words_file} に単語データが含まれていません。")
                return None

            segments_df = pd.read_csv(segments_file, encoding='utf-8-sig')
            segments = segments_df.to_dict('records')

            if not segments:
                print(f"[warning] {segments_file} にセグメントデータが含まれていません。")
                return None

            self.phase_progress_updated.emit(str(job.file_path), 2, 55.0, "話者埋め込み抽出中")

            config = asdict(job.settings_preset)

            self.phase_progress_updated.emit(str(job.file_path), 2, 80.0, "話者クラスタリング中")

            result = diarize_and_merge(audio_file, output_dir, config, words, segments)

            self.phase_progress_updated.emit(str(job.file_path), 2, 95.0, "話者ラベル統合中")

            if not result:
                if not has_token:
                    reason = "HUGGINGFACE_TOKENが設定されていないか、モデルがローカルキャッシュにありません。"
                else:
                    reason = "話者分離処理中にエラーが発生した可能性があります。"
                print(f"[warning] 話者分離がスキップされました。原因: {reason}")
                self.diarization_skipped.emit(str(job.file_path), reason)

            return result

        except Exception as e:
            error_msg = str(e)
            print(f"[warning] 話者分離がスキップされました。")
            print(f"[warning] エラー詳細: {error_msg}")
            import os
            token = os.getenv("HUGGINGFACE_TOKEN")
            if token:
                print(f"[warning] トークンは設定されていますが、認証に失敗した可能性があります。")
            import traceback
            print(f"[warning] 詳細なスタックトレース:")
            traceback.print_exc()
            self.diarization_skipped.emit(str(job.file_path), error_msg)
            return None

    def _convert_to_csv(self, output_dir: Path, stem: str, result: Dict[str, Any]):
        pass


class SettingsManager:
    def __init__(self):
        self.settings = QSettings("FWhisper", "BatchGUI")
        self.presets_dir = Path.home() / ".fwhisper_presets"
        self.presets_dir.mkdir(exist_ok=True)

    def save_preset(self, preset: SettingsPreset):
        preset_file = self.presets_dir / f"{preset.name}.json"
        with open(preset_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(preset), f, indent=2, ensure_ascii=False)

    def load_preset(self, name: str) -> Optional[SettingsPreset]:
        preset_file = self.presets_dir / f"{name}.json"
        if preset_file.exists():
            with open(preset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return SettingsPreset(**data)
        return None

    def list_presets(self) -> List[str]:
        return [f.stem for f in self.presets_dir.glob("*.json")]

    def delete_preset(self, name: str):
        preset_file = self.presets_dir / f"{name}.json"
        if preset_file.exists():
            preset_file.unlink()


class FileProgressWidget(QFrame):
    """1ファイルの処理進捗を表示するウィジェット"""
    remove_requested = Signal(str)  # file_path

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.setFrameStyle(QFrame.StyledPanel)
        self.setMaximumHeight(54)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(10)

        self.name_label = QLabel(Path(self.file_path).name)
        self.name_label.setMinimumWidth(180)
        self.name_label.setMaximumWidth(260)
        self.name_label.setToolTip(self.file_path)

        self.status_badge = QLabel("準備完了")
        self.status_badge.setMinimumWidth(160)
        self.status_badge.setMaximumWidth(190)
        self.status_badge.setAlignment(Qt.AlignCenter)
        self.status_badge.setStyleSheet(
            "border: 1px solid #ccc; border-radius: 3px; padding: 2px 6px; color: #666;"
        )

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #ddd; border-radius: 3px; background: #f5f5f5; }"
            "QProgressBar::chunk { background-color: #2196F3; border-radius: 2px; }"
        )

        self.time_label = QLabel("")
        self.time_label.setMinimumWidth(70)
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.time_label.setStyleSheet("color: #888; font-size: 11px;")

        self.remove_btn = QPushButton("×")
        self.remove_btn.setFixedSize(28, 28)
        self.remove_btn.setStyleSheet(
            "QPushButton { color: #999; border: 1px solid #ddd; border-radius: 3px; background: white; }"
            "QPushButton:hover:enabled { color: #f44336; border-color: #f44336; }"
            "QPushButton:disabled { color: #ccc; }"
        )
        self.remove_btn.clicked.connect(lambda: self.remove_requested.emit(self.file_path))

        layout.addWidget(self.name_label)
        layout.addWidget(self.status_badge)
        layout.addWidget(self.progress_bar, 1)
        layout.addWidget(self.time_label)
        layout.addWidget(self.remove_btn)

    def update_phase(self, phase_index: int, progress: float, phase_name: str):
        overall = int((phase_index * 100 + progress) / 4)
        self.progress_bar.setValue(overall)
        self.status_badge.setText(phase_name)
        self.status_badge.setStyleSheet(
            "border: 1px solid #1976D2; border-radius: 3px; padding: 2px 6px; "
            "color: #1976D2; background: #E3F2FD; font-weight: bold;"
        )
        self.remove_btn.setEnabled(False)

    def update_time(self, elapsed: float):
        self.time_label.setText(f"{elapsed:.1f}秒")

    def set_completed(self, time_str: str, diarization_skipped: bool = False):
        self.progress_bar.setValue(100)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #ddd; border-radius: 3px; background: #f5f5f5; }"
            "QProgressBar::chunk { background-color: #4CAF50; border-radius: 2px; }"
        )
        if diarization_skipped:
            self.status_badge.setText("完了 (話者分離省略)")
            self.status_badge.setStyleSheet(
                "border: 1px solid #F57C00; border-radius: 3px; padding: 2px 6px; "
                "color: #F57C00; background: #FFF3E0; font-weight: bold;"
            )
        else:
            self.status_badge.setText("完了")
            self.status_badge.setStyleSheet(
                "border: 1px solid #388E3C; border-radius: 3px; padding: 2px 6px; "
                "color: #388E3C; background: #E8F5E9; font-weight: bold;"
            )
        self.time_label.setText(time_str)
        self.remove_btn.setEnabled(True)

    def set_error(self, error: str = ""):
        self.status_badge.setText("エラー")
        self.status_badge.setStyleSheet(
            "border: 1px solid #D32F2F; border-radius: 3px; padding: 2px 6px; "
            "color: #D32F2F; background: #FFEBEE; font-weight: bold;"
        )
        self.status_badge.setToolTip(error)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #ddd; border-radius: 3px; background: #f5f5f5; }"
            "QProgressBar::chunk { background-color: #f44336; border-radius: 2px; }"
        )
        self.remove_btn.setEnabled(True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        load_env_file()

        self.settings_manager = SettingsManager()
        self.worker = TranscriptionWorker()
        self.current_preset = SettingsPreset(name="デフォルト")
        self.jobs: List[ProcessingJob] = []
        self.results_db = ResultsDatabase()
        self.file_widgets: Dict[str, FileProgressWidget] = {}

        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self.update_processing_time)
        self.current_processing_file = None
        self.processing_start_time = None

        self.setup_ui()
        self.setup_connections()
        self.load_last_preset()

    def setup_ui(self):
        self.setWindowTitle("KoeScribe - 音声文字起こし & 話者分離")
        self.setGeometry(100, 100, 1200, 800)

        icon_path = Path(__file__).parent.parent.parent / "app_icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # ── 左パネル（設定） ──
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)

        settings_group = QGroupBox("設定")
        settings_layout = QVBoxLayout(settings_group)

        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.setEditable(True)
        preset_row.addWidget(QLabel("プリセット:"))
        preset_row.addWidget(self.preset_combo)
        settings_layout.addLayout(preset_row)

        load_button_row = QHBoxLayout()
        self.load_preset_btn = QPushButton("読み込み")
        load_button_row.addWidget(self.load_preset_btn)
        load_button_row.addStretch()
        settings_layout.addLayout(load_button_row)

        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.model_combo.setCurrentText("large-v3")
        model_row.addWidget(QLabel("モデルサイズ:"))
        model_row.addWidget(self.model_combo)
        model_row.addWidget(self.create_help_button("model_size"))
        settings_layout.addLayout(model_row)

        lang_row = QHBoxLayout()
        self.language_combo = QComboBox()
        self.language_combo.addItems(["ja", "en", "auto"])
        self.language_combo.setCurrentText("ja")
        lang_row.addWidget(QLabel("言語:"))
        lang_row.addWidget(self.language_combo)
        settings_layout.addLayout(lang_row)

        device_row = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        self.device_combo.setCurrentText("auto")
        device_row.addWidget(QLabel("デバイス:"))
        device_row.addWidget(self.device_combo)
        settings_layout.addLayout(device_row)

        compute_row = QHBoxLayout()
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["auto", "float16", "int8_float16", "int8"])
        self.compute_combo.setCurrentText("auto")
        compute_row.addWidget(QLabel("計算精度:"))
        compute_row.addWidget(self.compute_combo)
        compute_row.addWidget(self.create_help_button("compute_type"))
        settings_layout.addLayout(compute_row)

        beam_row = QHBoxLayout()
        self.beam_size_spin = QSpinBox()
        self.beam_size_spin.setRange(1, 20)
        self.beam_size_spin.setValue(5)
        beam_row.addWidget(QLabel("ビームサイズ:"))
        beam_row.addWidget(self.beam_size_spin)
        beam_row.addWidget(self.create_help_button("beam_size"))
        settings_layout.addLayout(beam_row)

        vad_row = QHBoxLayout()
        self.vad_check = QCheckBox("VAD使用")
        self.vad_check.setChecked(True)
        vad_row.addWidget(self.vad_check)
        settings_layout.addLayout(vad_row)

        silence_row = QHBoxLayout()
        self.min_silence_spin = QSpinBox()
        self.min_silence_spin.setRange(100, 2000)
        self.min_silence_spin.setValue(500)
        self.min_silence_spin.setSuffix(" ms")
        silence_row.addWidget(QLabel("最小無音時間:"))
        silence_row.addWidget(self.min_silence_spin)
        silence_row.addWidget(self.create_help_button("min_silence_ms"))
        settings_layout.addLayout(silence_row)

        diarize_row = QHBoxLayout()
        self.diarization_check = QCheckBox("話者分離")
        self.diarization_check.setChecked(True)
        diarize_row.addWidget(self.diarization_check)
        settings_layout.addLayout(diarize_row)

        speakers_row = QHBoxLayout()
        self.max_speakers_spin = QSpinBox()
        self.max_speakers_spin.setRange(1, 20)
        self.max_speakers_spin.setValue(2)
        speakers_row.addWidget(QLabel("最大話者数:"))
        speakers_row.addWidget(self.max_speakers_spin)
        speakers_row.addWidget(self.create_help_button("max_speakers"))
        settings_layout.addLayout(speakers_row)

        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit("./outputs")
        self.output_dir_btn = QPushButton("参照")
        output_dir_layout.addWidget(QLabel("出力設定:"))
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_btn)
        settings_layout.addLayout(output_dir_layout)

        save_delete_row = QHBoxLayout()
        self.save_preset_btn = QPushButton("保存")
        self.delete_preset_btn = QPushButton("削除")
        save_delete_row.addWidget(self.save_preset_btn)
        save_delete_row.addWidget(self.delete_preset_btn)
        save_delete_row.addStretch()
        settings_layout.addLayout(save_delete_row)

        left_layout.addWidget(settings_group)
        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # ── 右パネル ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.tab_widget = QTabWidget()

        # 処理タブ
        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)
        processing_layout.setSpacing(8)

        # ファイル操作バー
        file_ctrl_bar = QHBoxLayout()
        self.add_files_btn = QPushButton("ファイル追加")
        self.add_folder_btn = QPushButton("フォルダ追加")
        self.open_output_btn = QPushButton("出力フォルダを開く")
        self.queue_label = QLabel("0件")
        self.queue_label.setStyleSheet("font-weight: bold; color: #666;")
        file_ctrl_bar.addWidget(self.add_files_btn)
        file_ctrl_bar.addWidget(self.add_folder_btn)
        file_ctrl_bar.addWidget(self.open_output_btn)
        file_ctrl_bar.addStretch()
        file_ctrl_bar.addWidget(self.queue_label)
        processing_layout.addLayout(file_ctrl_bar)

        # ファイルキュー（スクロールエリア）
        self.files_scroll = QScrollArea()
        self.files_scroll.setWidgetResizable(True)
        self.files_scroll.setFrameShape(QFrame.StyledPanel)

        self.files_container = QWidget()
        self.files_layout = QVBoxLayout(self.files_container)
        self.files_layout.setAlignment(Qt.AlignTop)
        self.files_layout.setSpacing(4)
        self.files_layout.setContentsMargins(4, 4, 4, 4)

        self.no_files_label = QLabel(
            "ファイルをドラッグ&ドロップするか\n「ファイル追加」ボタンで追加してください"
        )
        self.no_files_label.setAlignment(Qt.AlignCenter)
        self.no_files_label.setStyleSheet("color: #bbb; padding: 40px; font-size: 13px;")
        self.files_layout.addWidget(self.no_files_label)

        self.files_scroll.setWidget(self.files_container)
        processing_layout.addWidget(self.files_scroll)

        # 処理開始/停止ボタン
        bottom_bar = QHBoxLayout()
        self.start_btn = QPushButton("処理開始")
        self.start_btn.setEnabled(False)
        self.start_btn.setMinimumHeight(36)
        self.start_btn.setStyleSheet(
            "QPushButton:enabled { background-color: #4CAF50; color: white; font-weight: bold; font-size: 13px; border-radius: 4px; }"
            "QPushButton:disabled { background-color: #ccc; color: #888; border-radius: 4px; }"
        )
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(36)
        self.stop_btn.setStyleSheet(
            "QPushButton:enabled { background-color: #f44336; color: white; font-weight: bold; border-radius: 4px; }"
            "QPushButton:disabled { background-color: #ccc; color: #888; border-radius: 4px; }"
        )
        bottom_bar.addStretch()
        bottom_bar.addWidget(self.start_btn)
        bottom_bar.addWidget(self.stop_btn)
        processing_layout.addLayout(bottom_bar)

        # 別ファイル処理ボタン（処理完了後に表示）
        self.new_session_btn = QPushButton("別のファイルを処理する")
        self.new_session_btn.setVisible(False)
        self.new_session_btn.setMinimumHeight(40)
        self.new_session_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-size: 13px; "
            "font-weight: bold; border-radius: 4px; padding: 6px 20px;"
        )
        processing_layout.addWidget(self.new_session_btn)

        self.tab_widget.addTab(processing_tab, "処理")

        # 履歴タブ
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)

        history_buttons = QHBoxLayout()
        self.refresh_history_btn = QPushButton("履歴更新")
        self.clear_history_btn = QPushButton("履歴クリア")
        history_buttons.addWidget(self.refresh_history_btn)
        history_buttons.addWidget(self.clear_history_btn)
        history_buttons.addStretch()
        history_layout.addLayout(history_buttons)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(7)
        self.history_table.setHorizontalHeaderLabels(["ファイル名", "プリセット", "開始時刻", "状態", "処理時間", "言語", "音声長"])
        self.history_table.setColumnWidth(0, 140)
        self.history_table.setColumnWidth(1, 100)
        self.history_table.setColumnWidth(2, 130)
        self.history_table.setColumnWidth(3, 100)
        self.history_table.setColumnWidth(4, 80)
        self.history_table.setColumnWidth(5, 60)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        history_layout.addWidget(self.history_table)

        self.tab_widget.addTab(history_tab, "履歴")

        right_layout.addWidget(self.tab_widget)
        main_layout.addWidget(right_panel)

    def create_help_button(self, help_key):
        help_btn = QPushButton("?")
        help_btn.setFixedSize(20, 20)
        help_btn.setToolTip(HELP_TOOLTIPS.get(help_key, ""))
        help_btn.clicked.connect(lambda: QMessageBox.information(
            self, "ヘルプ", HELP_TOOLTIPS.get(help_key, "情報がありません")
        ))
        return help_btn

    def setup_connections(self):
        self.save_preset_btn.clicked.connect(self.save_preset)
        self.load_preset_btn.clicked.connect(self.load_preset)
        self.delete_preset_btn.clicked.connect(self.delete_preset)

        self.add_files_btn.clicked.connect(self.add_files)
        self.add_folder_btn.clicked.connect(self.add_folder)
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        self.open_output_btn.clicked.connect(self.open_output_directory)

        self.start_btn.clicked.connect(self.start_processing)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.new_session_btn.clicked.connect(self.reset_for_new_session)

        self.worker.progress_updated.connect(self.update_progress)
        self.worker.phase_progress_updated.connect(self.update_phase_progress)
        self.worker.timer_updated.connect(self.update_timer_display)
        self.worker.job_completed.connect(self.job_completed)
        self.worker.job_failed.connect(self.job_failed)
        self.worker.queue_updated.connect(self.update_queue_count)
        self.worker.diarization_skipped.connect(self.on_diarization_skipped)

        self.refresh_history_btn.clicked.connect(self.refresh_history)
        self.clear_history_btn.clicked.connect(self.clear_history)

        self.refresh_preset_list()

    def refresh_preset_list(self):
        current_text = self.preset_combo.currentText()
        self.preset_combo.clear()
        presets = self.settings_manager.list_presets()
        self.preset_combo.addItems(presets)
        if current_text:
            self.preset_combo.setCurrentText(current_text)

    def save_preset(self):
        preset = self.get_current_preset()
        if preset.name.strip():
            self.settings_manager.save_preset(preset)
            self.refresh_preset_list()
            QMessageBox.information(self, "保存完了", f"設定プリセット '{preset.name}' を保存しました。")
        else:
            QMessageBox.warning(self, "エラー", "プリセット名を入力してください。")

    def load_preset(self):
        name = self.preset_combo.currentText().strip()
        if name:
            preset = self.settings_manager.load_preset(name)
            if preset:
                self.apply_preset(preset)
                QMessageBox.information(self, "読み込み完了", f"設定プリセット '{name}' を読み込みました。")
            else:
                QMessageBox.warning(self, "エラー", f"設定プリセット '{name}' が見つかりません。")

    def delete_preset(self):
        name = self.preset_combo.currentText().strip()
        if name:
            reply = QMessageBox.question(
                self, "確認", f"設定プリセット '{name}' を削除しますか？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.settings_manager.delete_preset(name)
                self.refresh_preset_list()
                QMessageBox.information(self, "削除完了", f"設定プリセット '{name}' を削除しました。")

    def get_current_preset(self) -> SettingsPreset:
        return SettingsPreset(
            name=self.preset_combo.currentText().strip(),
            model_size=self.model_combo.currentText(),
            language=self.language_combo.currentText(),
            device=self.device_combo.currentText(),
            compute_type=self.compute_combo.currentText(),
            beam_size=self.beam_size_spin.value(),
            use_vad=self.vad_check.isChecked(),
            min_silence_ms=self.min_silence_spin.value(),
            enable_diarization=self.diarization_check.isChecked(),
            max_speakers=self.max_speakers_spin.value(),
            output_dir=self.output_dir_edit.text()
        )

    def apply_preset(self, preset: SettingsPreset):
        self.model_combo.setCurrentText(preset.model_size)
        self.language_combo.setCurrentText(preset.language)
        self.device_combo.setCurrentText(preset.device)
        self.compute_combo.setCurrentText(preset.compute_type)
        self.beam_size_spin.setValue(preset.beam_size)
        self.vad_check.setChecked(preset.use_vad)
        self.min_silence_spin.setValue(preset.min_silence_ms)
        self.diarization_check.setChecked(preset.enable_diarization)
        self.max_speakers_spin.setValue(preset.max_speakers)
        self.output_dir_edit.setText(preset.output_dir)

    def load_last_preset(self):
        # まず「デフォルト」を試し、なければ一番最初のプリセットを読み込む
        default_preset = self.settings_manager.load_preset("デフォルト")
        if default_preset:
            self.apply_preset(default_preset)
            return
        presets = self.settings_manager.list_presets()
        if presets:
            first_preset = self.settings_manager.load_preset(presets[0])
            if first_preset:
                self.apply_preset(first_preset)
                self.preset_combo.setCurrentText(presets[0])

    def _add_file_widget(self, file_path: str):
        """ファイルウィジェットをスクロールエリアに追加"""
        if file_path in self.file_widgets:
            return

        # 「ファイルなし」ラベルを非表示
        self.no_files_label.setVisible(False)

        widget = FileProgressWidget(file_path)
        widget.remove_requested.connect(self.remove_file_widget)
        self.files_layout.addWidget(widget)
        self.file_widgets[file_path] = widget

    def remove_file_widget(self, file_path: str):
        """ファイルウィジェットを削除"""
        if file_path not in self.file_widgets:
            return

        widget = self.file_widgets.pop(file_path)
        self.files_layout.removeWidget(widget)
        widget.deleteLater()

        # ジョブリストからも削除
        self.jobs = [j for j in self.jobs if str(j.file_path) != file_path]

        # ファイルが0件になったら「ファイルなし」ラベルを表示
        if not self.file_widgets:
            self.no_files_label.setVisible(True)
            self.start_btn.setEnabled(False)

        self.update_queue_display()

    def add_files_to_queue(self, file_paths: List[Path], add_to_list: bool = True):
        preset = self.get_current_preset()
        added_count = 0
        skipped_count = 0

        existing_paths = set(str(j.file_path) for j in self.jobs)
        if self.worker.current_job:
            existing_paths.add(str(self.worker.current_job.file_path))

        for file_path in file_paths:
            file_path_str = str(file_path)

            if file_path_str in existing_paths:
                skipped_count += 1
                continue

            job = ProcessingJob(file_path, preset)
            self.jobs.append(job)
            existing_paths.add(file_path_str)

            if add_to_list:
                self._add_file_widget(file_path_str)

            added_count += 1

        if added_count > 0:
            self.start_btn.setEnabled(True)
            self.new_session_btn.setVisible(False)
            self.update_queue_display()

        return added_count, skipped_count

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "音声・動画ファイルを選択",
            "", "Media Files (*.wav *.mp4 *.mkv *.avi *.mov *.flv *.wmv *.webm *.m4v *.mp3 *.m4a *.flac *.ogg);;All Files (*.*)"
        )

        if files:
            file_paths = [Path(f) for f in files]
            added, skipped = self.add_files_to_queue(file_paths, add_to_list=True)

            if skipped > 0:
                QMessageBox.information(
                    self, "ファイル追加",
                    f"{added}件のファイルを追加しました。\n{skipped}件は既に追加済みです。"
                )

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "フォルダを選択")
        if folder:
            folder_path = Path(folder)
            extensions = ['.wav', '.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm', '.m4v', '.mp3', '.m4a', '.flac', '.ogg']

            file_paths = []
            for ext in extensions:
                for file_path in folder_path.glob(f"*{ext}"):
                    file_paths.append(file_path)

            if file_paths:
                added, skipped = self.add_files_to_queue(file_paths, add_to_list=True)
                msg = f"{added}件のファイルを追加しました。"
                if skipped > 0:
                    msg += f"\n{skipped}件は既に追加済みです。"
                QMessageBox.information(self, "フォルダ追加", msg)
            else:
                QMessageBox.information(self, "フォルダ追加", "処理可能なファイルが見つかりませんでした。")

    def select_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "出力フォルダを選択")
        if folder:
            self.output_dir_edit.setText(folder)

    def open_output_directory(self):
        output_dir = Path(self.output_dir_edit.text())
        if output_dir.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_dir.absolute())))
        else:
            QMessageBox.warning(self, "警告", "出力ディレクトリが存在しません。")

    def start_processing(self):
        if len(self.jobs) == 0:
            QMessageBox.warning(self, "警告", "処理するファイルを追加してください。")
            return

        if self.worker.isRunning():
            QMessageBox.information(self, "処理中", "既に処理が実行中です。")
            return

        for job in self.jobs:
            self.worker.add_job(job)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.stop_btn.setVisible(True)
        self.new_session_btn.setVisible(False)

        if self.jobs:
            self.start_processing_timer(str(self.jobs[0].file_path))

        self.worker.start()

    def stop_processing(self):
        self.worker.stop_processing()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def reset_for_new_session(self):
        """ファイルリストとジョブをクリアして初期状態に戻す（設定は保持）"""
        # ウィジェットをすべて削除
        for file_path, widget in list(self.file_widgets.items()):
            self.files_layout.removeWidget(widget)
            widget.deleteLater()
        self.file_widgets.clear()
        self.jobs.clear()

        self.no_files_label.setVisible(True)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.new_session_btn.setVisible(False)
        self.update_queue_display()

    def update_progress(self, file_path: str, progress: float, stage: str):
        widget = self.file_widgets.get(file_path)
        if widget:
            widget.status_badge.setText(stage)

    def update_phase_progress(self, file_path: str, phase_index: int, progress: float, phase_name: str):
        widget = self.file_widgets.get(file_path)
        if widget:
            widget.update_phase(phase_index, progress, phase_name)

    def update_queue_count(self, count: int):
        self.update_queue_display()

    def update_queue_display(self):
        total_jobs = len(self.jobs)

        if self.worker.current_job:
            current_index = 1
            for i, job in enumerate(self.jobs):
                if str(job.file_path) == str(self.worker.current_job.file_path):
                    current_index = i + 1
                    break
            self.queue_label.setText(f"{current_index}件目/{total_jobs}件")
        elif total_jobs == 0:
            self.queue_label.setText("0件")
        else:
            self.queue_label.setText(f"{total_jobs}件")

    def job_completed(self, file_path: str, result: Dict[str, Any]):
        if self.current_processing_file == file_path:
            self.stop_processing_timer()

        job_preset = None
        processing_time = result.get('processing_time', 0)
        for job in self.jobs:
            if str(job.file_path) == file_path:
                job_preset = job.settings_preset
                break

        diarization_skipped = False
        if job_preset:
            diarization_skipped = job_preset.enable_diarization and not result.get('diarization')

        widget = self.file_widgets.get(file_path)
        if widget:
            widget.set_completed(f"{processing_time:.1f}秒", diarization_skipped=diarization_skipped)

        self.update_queue_display()

        # すべて完了したか確認
        queue_size = self.worker.jobs.qsize()

        if queue_size == 0 and not self.worker.isRunning():
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.new_session_btn.setVisible(True)
            self.refresh_history()
            self.update_queue_display()
        elif queue_size > 0:
            try:
                temp_jobs = []
                next_job = None
                while not self.worker.jobs.empty():
                    job = self.worker.jobs.get_nowait()
                    temp_jobs.append(job)
                    if next_job is None:
                        next_job = job
                for job in temp_jobs:
                    self.worker.jobs.put(job)
                if next_job:
                    self.start_processing_timer(str(next_job.file_path))
            except Exception as e:
                print(f"Error checking next job: {e}")

    def job_failed(self, file_path: str, error: str):
        if self.current_processing_file == file_path:
            self.stop_processing_timer()

        widget = self.file_widgets.get(file_path)
        if widget:
            widget.set_error(error)

        # すべて完了したか確認（エラーも完了扱い）
        queue_size = self.worker.jobs.qsize()
        if queue_size == 0 and not self.worker.isRunning():
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.new_session_btn.setVisible(True)
            self.refresh_history()

        QMessageBox.warning(
            self, "処理エラー",
            f"ファイル '{Path(file_path).name}' の処理中にエラーが発生しました:\n{error}"
        )

    def on_diarization_skipped(self, file_path: str, reason: str):
        """話者分離スキップ - ウィジェットのステータスで表示するのみ（ポップアップなし）"""
        pass

    def refresh_history(self):
        results = self.results_db.get_recent_results()
        self.history_table.setRowCount(len(results))

        for i, result in enumerate(results):
            self.history_table.setItem(i, 0, QTableWidgetItem(Path(result['file_path']).name))
            self.history_table.setItem(i, 1, QTableWidgetItem(result['project_name'] or ""))
            self.history_table.setItem(i, 2, QTableWidgetItem(str(result['start_time'])[:19]))
            self.history_table.setItem(i, 3, QTableWidgetItem(result['status']))
            self.history_table.setItem(i, 4, QTableWidgetItem(f"{result['processing_time']:.2f}秒" if result['processing_time'] else ""))
            self.history_table.setItem(i, 5, QTableWidgetItem(result['estimated_language'] or ""))
            self.history_table.setItem(i, 6, QTableWidgetItem(f"{result['audio_duration']:.2f}秒" if result['audio_duration'] else ""))

    def clear_history(self):
        reply = QMessageBox.question(
            self, "確認", "処理履歴をすべて削除しますか？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            conn = sqlite3.connect(self.results_db.db_path)
            conn.execute("DELETE FROM processing_results")
            conn.commit()
            conn.close()
            self.refresh_history()
            QMessageBox.information(self, "削除完了", "処理履歴を削除しました。")

    def update_timer_display(self, file_path: str, elapsed_seconds: float):
        widget = self.file_widgets.get(file_path)
        if widget:
            widget.update_time(elapsed_seconds)

    def update_processing_time(self):
        if self.current_processing_file and self.processing_start_time:
            elapsed = time.time() - self.processing_start_time
            self.update_timer_display(self.current_processing_file, elapsed)

    def start_processing_timer(self, file_path: str):
        self.current_processing_file = file_path
        self.processing_start_time = time.time()
        self.processing_timer.start(500)

    def stop_processing_timer(self):
        self.processing_timer.stop()
        self.current_processing_file = None
        self.processing_start_time = None


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("KoeScribe")
    app.setOrganizationName("KoeScribe")

    window = MainWindow()
    window.show()

    if not VideoConverter.is_ffmpeg_available():
        QMessageBox.warning(
            window, "FFmpeg未検出",
            "FFmpegがインストールされていないか、PATHに設定されていません。\n\n"
            "動画ファイルの変換機能は使用できません。WAVファイルのみ処理可能です。\n\n"
            "動画ファイルを処理したい場合は、FFmpegをインストールしてください:\n"
            "https://ffmpeg.org/download.html"
        )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
