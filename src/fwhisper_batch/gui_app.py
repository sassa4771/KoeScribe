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
    QLabel, QPushButton, QFileDialog, QProgressBar, QListWidget,
    QListWidgetItem, QGroupBox, QSpinBox, QLineEdit, QComboBox,
    QTextEdit, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QSplitter, QFormLayout, QCheckBox
)
from PySide6.QtCore import QThread, Signal, QTimer, Qt, QSettings, QUrl
from PySide6.QtGui import QFont, QIcon, QDesktopServices

import pandas as pd
from faster_whisper import WhisperModel

from .transcribe_batch import (
    transcribe_one, load_config, resolve_device, resolve_compute_type,
    detect_device, is_diarization_enabled
)


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
    diarize_model: str = "pyannote/speaker-diarization"
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
                speaker_count INTEGER,
                output_directory TEXT,
                error_message TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def save_result(self, job: ProcessingJob) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO processing_results 
            (file_path, project_name, start_time, end_time, status, processing_time, speaker_count, output_directory, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(job.file_path),
            job.settings_preset.name,
            datetime.now(),
            datetime.now(),
            job.status,
            job.result.get('processing_time', 0) if job.result else 0,
            job.result.get('diarization', {}).get('segments_with_speakers', 0) if job.result and job.result.get('diarization') else 0,
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
    job_completed = Signal(str, dict)
    job_failed = Signal(str, str)
    queue_updated = Signal(int)
    
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
        
    def run(self):
        while not self.should_stop:
            try:
                job = self.jobs.get(timeout=1.0)
                self.current_job = job
                self.process_job(job)
                self.jobs.task_done()
                self.queue_updated.emit(self.jobs.qsize())
            except queue.Empty:
                if self.jobs.empty():
                    break
                continue
        
        self.current_job = None
        
    def process_job(self, job: ProcessingJob):
        try:
            self.progress_updated.emit(str(job.file_path), 0.0, "モデル読み込み中...")
            
            if self.model is None:
                device = resolve_device(job.settings_preset.device)
                compute_type = resolve_compute_type(device, job.settings_preset.compute_type)
                self.model = WhisperModel(
                    job.settings_preset.model_size,
                    device=device,
                    compute_type=compute_type
                )
            
            self.progress_updated.emit(str(job.file_path), 20.0, "音声解析中...")
            
            output_dir = Path(job.settings_preset.output_dir) / f"output_{job.file_path.stem}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result = transcribe_one(
                self.model,
                job.file_path,
                output_dir,
                job.settings_preset.language,
                job.settings_preset.beam_size,
                job.settings_preset.use_vad,
                job.settings_preset.min_silence_ms,
                word_timestamps=True,
                show_progress=False,
                enable_diarization=job.settings_preset.enable_diarization,
                diarization_config=asdict(job.settings_preset) if job.settings_preset.enable_diarization else None
            )
            
            self.progress_updated.emit(str(job.file_path), 90.0, "CSV変換中...")
            
            self._convert_to_csv(output_dir, job.file_path.stem, result)
            
            job.result = result
            job.status = "完了"
            job.job_id = self.results_db.save_result(job)
            
            self.progress_updated.emit(str(job.file_path), 100.0, "完了")
            self.job_completed.emit(str(job.file_path), result)
            
        except Exception as e:
            job.error = str(e)
            job.status = "エラー"
            job.job_id = self.results_db.save_result(job)
            self.job_failed.emit(str(job.file_path), str(e))
            
    def _convert_to_csv(self, output_dir: Path, stem: str, result: Dict[str, Any]):
        segments_file = output_dir / f"{stem}_segments_with_speakers.jsonl"
        if segments_file.exists():
            segments_data = []
            with open(segments_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        segments_data.append(json.loads(line))
            
            if segments_data:
                df = pd.DataFrame(segments_data)
                csv_file = output_dir / f"{stem}_segments_with_speakers.csv"
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        words_file = output_dir / f"{stem}_words_with_speakers.jsonl"
        if words_file.exists():
            words_data = []
            with open(words_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        words_data.append(json.loads(line))
            
            if words_data:
                df = pd.DataFrame(words_data)
                csv_file = output_dir / f"{stem}_words_with_speakers.csv"
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')


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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings_manager = SettingsManager()
        self.worker = TranscriptionWorker()
        self.current_preset = SettingsPreset(name="デフォルト")
        self.jobs: List[ProcessingJob] = []
        self.results_db = ResultsDatabase()
        
        self.setup_ui()
        self.setup_connections()
        self.load_last_preset()
        
    def setup_ui(self):
        self.setWindowTitle("FWhisper Batch GUI v2 - 音声文字起こし & 話者分離")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
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
        
        left_layout.addWidget(settings_group)
        
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
        
        main_layout.addWidget(left_panel)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.tab_widget = QTabWidget()
        
        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)
        
        file_group = QGroupBox("ファイル選択 (WAVファイルのみ)")
        file_layout = QVBoxLayout(file_group)
        
        file_buttons = QHBoxLayout()
        self.add_files_btn = QPushButton("WAVファイル追加")
        self.add_folder_btn = QPushButton("フォルダ追加")
        self.clear_files_btn = QPushButton("クリア")
        file_buttons.addWidget(self.add_files_btn)
        file_buttons.addWidget(self.add_folder_btn)
        file_buttons.addWidget(self.clear_files_btn)
        file_layout.addLayout(file_buttons)
        
        self.file_list = QListWidget()
        file_layout.addWidget(self.file_list)
        
        processing_layout.addWidget(file_group)
        
        progress_group = QGroupBox("処理状況")
        progress_layout = QVBoxLayout(progress_group)
        
        self.current_file_label = QLabel("待機中")
        self.progress_bar = QProgressBar()
        self.stage_label = QLabel("")
        self.queue_label = QLabel("キュー: 0件")
        
        progress_layout.addWidget(self.current_file_label)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.stage_label)
        progress_layout.addWidget(self.queue_label)
        
        control_buttons = QHBoxLayout()
        self.start_btn = QPushButton("処理開始")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        control_buttons.addWidget(self.start_btn)
        control_buttons.addWidget(self.stop_btn)
        progress_layout.addLayout(control_buttons)
        
        processing_layout.addWidget(progress_group)
        
        results_group = QGroupBox("処理結果")
        results_layout = QVBoxLayout(results_group)
        
        results_buttons = QHBoxLayout()
        self.open_output_btn = QPushButton("📁 出力フォルダを開く")
        results_buttons.addWidget(self.open_output_btn)
        results_buttons.addStretch()
        results_layout.addLayout(results_buttons)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["ファイル名", "状態", "処理時間", "出力"])
        
        self.results_table.setColumnWidth(0, 200)  # ファイル名
        self.results_table.setColumnWidth(1, 150)  # 状態（幅を広げる）
        self.results_table.setColumnWidth(2, 100)  # 処理時間
        self.results_table.horizontalHeader().setStretchLastSection(True)  # 出力列は残りスペースを使用
        
        results_layout.addWidget(self.results_table)
        
        processing_layout.addWidget(results_group)
        
        self.tab_widget.addTab(processing_tab, "処理")
        
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
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels(["ファイル名", "プリセット", "開始時刻", "状態", "処理時間", "話者数"])
        
        self.history_table.setColumnWidth(0, 150)  # ファイル名
        self.history_table.setColumnWidth(1, 120)  # プリセット
        self.history_table.setColumnWidth(2, 140)  # 開始時刻
        self.history_table.setColumnWidth(3, 120)  # 状態（幅を広げる）
        self.history_table.setColumnWidth(4, 100)  # 処理時間
        self.history_table.horizontalHeader().setStretchLastSection(True)  # 話者数列は残りスペースを使用
        
        history_layout.addWidget(self.history_table)
        
        self.tab_widget.addTab(history_tab, "履歴")
        
        right_layout.addWidget(self.tab_widget)
        
        main_layout.addWidget(right_panel)
        
    def create_help_button(self, help_key):
        """Create a help button for settings"""
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
        self.clear_files_btn.clicked.connect(self.clear_files)
        
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        self.open_output_btn.clicked.connect(self.open_output_directory)
        
        self.start_btn.clicked.connect(self.start_processing)
        self.stop_btn.clicked.connect(self.stop_processing)
        
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.job_completed.connect(self.job_completed)
        self.worker.job_failed.connect(self.job_failed)
        self.worker.queue_updated.connect(self.update_queue_count)
        
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
        default_preset = self.settings_manager.load_preset("デフォルト")
        if default_preset:
            self.apply_preset(default_preset)
            
    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "WAVファイルを選択",
            "", "WAV Files (*.wav)"
        )
        
        for file_path in files:
            item = QListWidgetItem(file_path)
            self.file_list.addItem(item)
            
        if self.worker.isRunning():
            preset = self.get_current_preset()
            for file_path in files:
                job = ProcessingJob(Path(file_path), preset)
                self.worker.add_job(job)
                
    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "フォルダを選択")
        if folder:
            folder_path = Path(folder)
            extensions = ['.wav']
            
            new_files = []
            for ext in extensions:
                for file_path in folder_path.glob(f"*{ext}"):
                    item = QListWidgetItem(str(file_path))
                    self.file_list.addItem(item)
                    new_files.append(str(file_path))
                    
            if self.worker.isRunning() and new_files:
                preset = self.get_current_preset()
                for file_path in new_files:
                    job = ProcessingJob(Path(file_path), preset)
                    self.worker.add_job(job)
                    
    def clear_files(self):
        self.file_list.clear()
        
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
        if self.file_list.count() == 0:
            QMessageBox.warning(self, "警告", "処理するファイルを選択してください。")
            return
            
        preset = self.get_current_preset()
        self.jobs.clear()
        
        for i in range(self.file_list.count()):
            file_path = Path(self.file_list.item(i).text())
            job = ProcessingJob(file_path, preset)
            self.jobs.append(job)
            self.worker.add_job(job)
            
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.results_table.setRowCount(len(self.jobs))
        for i, job in enumerate(self.jobs):
            self.results_table.setItem(i, 0, QTableWidgetItem(job.file_path.name))
            self.results_table.setItem(i, 1, QTableWidgetItem("待機中"))
            self.results_table.setItem(i, 2, QTableWidgetItem(""))
            self.results_table.setItem(i, 3, QTableWidgetItem(""))
            
        self.worker.start()
        
    def stop_processing(self):
        self.worker.stop_processing()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
    def update_progress(self, file_path: str, progress: float, stage: str):
        self.current_file_label.setText(f"処理中: {Path(file_path).name}")
        self.progress_bar.setValue(int(progress))
        self.stage_label.setText(stage)
        
        for i in range(self.results_table.rowCount()):
            if self.results_table.item(i, 0).text() == Path(file_path).name:
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{stage} ({progress:.1f}%)"))
                break
                
    def update_queue_count(self, count: int):
        self.queue_label.setText(f"キュー: {count}件")
        
    def job_completed(self, file_path: str, result: Dict[str, Any]):
        file_name = Path(file_path).name
        
        for i in range(self.results_table.rowCount()):
            if self.results_table.item(i, 0).text() == file_name:
                self.results_table.setItem(i, 1, QTableWidgetItem("完了"))
                self.results_table.setItem(i, 2, QTableWidgetItem(f"{result.get('processing_time', 0):.2f}秒"))
                
                output_dir = Path(self.get_current_preset().output_dir) / f"output_{Path(file_path).stem}"
                csv_files = list(output_dir.glob("*.csv"))
                if csv_files:
                    self.results_table.setItem(i, 3, QTableWidgetItem(f"{len(csv_files)} CSVファイル"))
                break
                
        if not self.worker.isRunning() and self.worker.jobs.empty():
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.current_file_label.setText("すべての処理が完了しました")
            self.progress_bar.setValue(100)
            self.stage_label.setText("完了")
            self.refresh_history()
            
    def job_failed(self, file_path: str, error: str):
        file_name = Path(file_path).name
        
        for i in range(self.results_table.rowCount()):
            if self.results_table.item(i, 0).text() == file_name:
                self.results_table.setItem(i, 1, QTableWidgetItem("エラー"))
                self.results_table.setItem(i, 3, QTableWidgetItem(error))
                break
                
        QMessageBox.warning(self, "処理エラー", f"ファイル '{file_name}' の処理中にエラーが発生しました:\n{error}")
        
    def refresh_history(self):
        results = self.results_db.get_recent_results()
        self.history_table.setRowCount(len(results))
        
        for i, result in enumerate(results):
            self.history_table.setItem(i, 0, QTableWidgetItem(Path(result['file_path']).name))
            self.history_table.setItem(i, 1, QTableWidgetItem(result['project_name'] or ""))
            self.history_table.setItem(i, 2, QTableWidgetItem(str(result['start_time'])[:19]))
            self.history_table.setItem(i, 3, QTableWidgetItem(result['status']))
            self.history_table.setItem(i, 4, QTableWidgetItem(f"{result['processing_time']:.2f}秒" if result['processing_time'] else ""))
            self.history_table.setItem(i, 5, QTableWidgetItem(str(result['speaker_count']) if result['speaker_count'] else ""))
            
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


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FWhisper Batch GUI v2")
    app.setOrganizationName("FWhisper")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
