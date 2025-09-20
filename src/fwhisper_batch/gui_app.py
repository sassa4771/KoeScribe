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
class ProjectConfig:
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
    project_config: ProjectConfig
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
            job.project_config.name,
            datetime.now(),
            datetime.now(),
            job.status,
            job.result.get('processing_time', 0) if job.result else 0,
            job.result.get('diarization', {}).get('segments_with_speakers', 0) if job.result and job.result.get('diarization') else 0,
            job.project_config.output_dir,
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
                device = resolve_device(job.project_config.device)
                compute_type = resolve_compute_type(device, job.project_config.compute_type)
                self.model = WhisperModel(
                    job.project_config.model_size,
                    device=device,
                    compute_type=compute_type
                )
            
            self.progress_updated.emit(str(job.file_path), 20.0, "音声解析中...")
            
            output_dir = Path(job.project_config.output_dir) / f"output_{job.file_path.stem}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result = transcribe_one(
                self.model,
                job.file_path,
                output_dir,
                job.project_config.language,
                job.project_config.beam_size,
                job.project_config.use_vad,
                job.project_config.min_silence_ms,
                word_timestamps=True,
                show_progress=False,
                enable_diarization=job.project_config.enable_diarization,
                diarization_config=asdict(job.project_config) if job.project_config.enable_diarization else None
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


class ProjectManager:
    def __init__(self):
        self.settings = QSettings("FWhisper", "BatchGUI")
        self.projects_dir = Path.home() / ".fwhisper_projects"
        self.projects_dir.mkdir(exist_ok=True)
        
    def save_project(self, config: ProjectConfig):
        project_file = self.projects_dir / f"{config.name}.json"
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)
            
    def load_project(self, name: str) -> Optional[ProjectConfig]:
        project_file = self.projects_dir / f"{name}.json"
        if project_file.exists():
            with open(project_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return ProjectConfig(**data)
        return None
        
    def list_projects(self) -> List[str]:
        return [f.stem for f in self.projects_dir.glob("*.json")]
        
    def delete_project(self, name: str):
        project_file = self.projects_dir / f"{name}.json"
        if project_file.exists():
            project_file.unlink()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project_manager = ProjectManager()
        self.worker = TranscriptionWorker()
        self.current_config = ProjectConfig(name="デフォルト")
        self.jobs: List[ProcessingJob] = []
        self.results_db = ResultsDatabase()
        
        self.setup_ui()
        self.setup_connections()
        self.load_last_project()
        
    def setup_ui(self):
        self.setWindowTitle("FWhisper Batch GUI v2 - 音声文字起こし & 話者分離")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        
        project_group = QGroupBox("プロジェクト管理")
        project_layout = QVBoxLayout(project_group)
        
        project_row = QHBoxLayout()
        self.project_combo = QComboBox()
        self.project_combo.setEditable(True)
        project_row.addWidget(QLabel("プロジェクト:"))
        project_row.addWidget(self.project_combo)
        project_layout.addLayout(project_row)
        
        project_buttons = QHBoxLayout()
        self.save_project_btn = QPushButton("保存")
        self.load_project_btn = QPushButton("読み込み")
        self.delete_project_btn = QPushButton("削除")
        project_buttons.addWidget(self.save_project_btn)
        project_buttons.addWidget(self.load_project_btn)
        project_buttons.addWidget(self.delete_project_btn)
        project_layout.addLayout(project_buttons)
        
        left_layout.addWidget(project_group)
        
        settings_group = QGroupBox("設定")
        settings_layout = QFormLayout(settings_group)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.model_combo.setCurrentText("large-v3")
        self.add_help_widget(settings_layout, "モデルサイズ:", self.model_combo, "model_size")
        
        self.language_combo = QComboBox()
        self.language_combo.addItems(["ja", "en", "auto"])
        self.language_combo.setCurrentText("ja")
        self.add_help_widget(settings_layout, "言語:", self.language_combo, "language")
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        self.device_combo.setCurrentText("auto")
        self.add_help_widget(settings_layout, "デバイス:", self.device_combo, "device")
        
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["auto", "float16", "int8_float16", "int8"])
        self.compute_combo.setCurrentText("auto")
        self.add_help_widget(settings_layout, "計算精度:", self.compute_combo, "compute_type")
        
        self.beam_size_spin = QSpinBox()
        self.beam_size_spin.setRange(1, 20)
        self.beam_size_spin.setValue(5)
        self.add_help_widget(settings_layout, "ビームサイズ:", self.beam_size_spin, "beam_size")
        
        self.vad_check = QCheckBox()
        self.vad_check.setChecked(True)
        self.add_help_widget(settings_layout, "VAD使用:", self.vad_check, "use_vad")
        
        self.min_silence_spin = QSpinBox()
        self.min_silence_spin.setRange(100, 2000)
        self.min_silence_spin.setValue(500)
        self.min_silence_spin.setSuffix(" ms")
        self.add_help_widget(settings_layout, "最小無音時間:", self.min_silence_spin, "min_silence_ms")
        
        self.diarization_check = QCheckBox()
        self.diarization_check.setChecked(True)
        settings_layout.addRow("話者分離:", self.diarization_check)
        
        self.max_speakers_spin = QSpinBox()
        self.max_speakers_spin.setRange(1, 20)
        self.max_speakers_spin.setValue(2)
        self.add_help_widget(settings_layout, "最大話者数:", self.max_speakers_spin, "max_speakers")
        
        left_layout.addWidget(settings_group)
        
        output_group = QGroupBox("出力設定")
        output_layout = QVBoxLayout(output_group)
        
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit("./outputs")
        self.output_dir_btn = QPushButton("参照")
        self.open_output_btn = QPushButton("📁 出力フォルダを開く")
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_dir_btn)
        output_layout.addLayout(output_dir_layout)
        output_layout.addWidget(self.open_output_btn)
        
        left_layout.addWidget(output_group)
        
        main_layout.addWidget(left_panel)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.tab_widget = QTabWidget()
        
        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)
        
        file_group = QGroupBox("ファイル選択")
        file_layout = QVBoxLayout(file_group)
        
        file_buttons = QHBoxLayout()
        self.add_files_btn = QPushButton("ファイル追加")
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
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["ファイル名", "状態", "処理時間", "出力"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
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
        self.history_table.setHorizontalHeaderLabels(["ファイル名", "プロジェクト", "開始時刻", "状態", "処理時間", "話者数"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        history_layout.addWidget(self.history_table)
        
        self.tab_widget.addTab(history_tab, "履歴")
        
        right_layout.addWidget(self.tab_widget)
        
        main_layout.addWidget(right_panel)
        
    def add_help_widget(self, layout, label_text, widget, help_key):
        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        container_layout.addWidget(widget)
        
        help_btn = QPushButton("?")
        help_btn.setFixedSize(20, 20)
        help_btn.setToolTip(HELP_TOOLTIPS.get(help_key, ""))
        help_btn.clicked.connect(lambda: QMessageBox.information(
            self, "ヘルプ", HELP_TOOLTIPS.get(help_key, "情報がありません")
        ))
        container_layout.addWidget(help_btn)
        
        layout.addRow(label_text, container)
        
    def setup_connections(self):
        self.save_project_btn.clicked.connect(self.save_project)
        self.load_project_btn.clicked.connect(self.load_project)
        self.delete_project_btn.clicked.connect(self.delete_project)
        
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
        
        self.refresh_project_list()
        
    def refresh_project_list(self):
        current_text = self.project_combo.currentText()
        self.project_combo.clear()
        projects = self.project_manager.list_projects()
        self.project_combo.addItems(projects)
        if current_text:
            self.project_combo.setCurrentText(current_text)
            
    def save_project(self):
        config = self.get_current_config()
        if config.name.strip():
            self.project_manager.save_project(config)
            self.refresh_project_list()
            QMessageBox.information(self, "保存完了", f"プロジェクト '{config.name}' を保存しました。")
        else:
            QMessageBox.warning(self, "エラー", "プロジェクト名を入力してください。")
            
    def load_project(self):
        name = self.project_combo.currentText().strip()
        if name:
            config = self.project_manager.load_project(name)
            if config:
                self.apply_config(config)
                QMessageBox.information(self, "読み込み完了", f"プロジェクト '{name}' を読み込みました。")
            else:
                QMessageBox.warning(self, "エラー", f"プロジェクト '{name}' が見つかりません。")
                
    def delete_project(self):
        name = self.project_combo.currentText().strip()
        if name:
            reply = QMessageBox.question(
                self, "確認", f"プロジェクト '{name}' を削除しますか？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.project_manager.delete_project(name)
                self.refresh_project_list()
                QMessageBox.information(self, "削除完了", f"プロジェクト '{name}' を削除しました。")
                
    def get_current_config(self) -> ProjectConfig:
        return ProjectConfig(
            name=self.project_combo.currentText().strip(),
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
        
    def apply_config(self, config: ProjectConfig):
        self.model_combo.setCurrentText(config.model_size)
        self.language_combo.setCurrentText(config.language)
        self.device_combo.setCurrentText(config.device)
        self.compute_combo.setCurrentText(config.compute_type)
        self.beam_size_spin.setValue(config.beam_size)
        self.vad_check.setChecked(config.use_vad)
        self.min_silence_spin.setValue(config.min_silence_ms)
        self.diarization_check.setChecked(config.enable_diarization)
        self.max_speakers_spin.setValue(config.max_speakers)
        self.output_dir_edit.setText(config.output_dir)
        
    def load_last_project(self):
        default_config = self.project_manager.load_project("デフォルト")
        if default_config:
            self.apply_config(default_config)
            
    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "音声・動画ファイルを選択",
            "", "Audio/Video Files (*.wav *.mp3 *.m4a *.mp4 *.avi *.mov *.flv)"
        )
        
        for file_path in files:
            item = QListWidgetItem(file_path)
            self.file_list.addItem(item)
            
        if self.worker.isRunning():
            config = self.get_current_config()
            for file_path in files:
                job = ProcessingJob(Path(file_path), config)
                self.worker.add_job(job)
                
    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "フォルダを選択")
        if folder:
            folder_path = Path(folder)
            extensions = ['.wav', '.mp3', '.m4a', '.mp4', '.avi', '.mov', '.flv']
            
            new_files = []
            for ext in extensions:
                for file_path in folder_path.glob(f"*{ext}"):
                    item = QListWidgetItem(str(file_path))
                    self.file_list.addItem(item)
                    new_files.append(str(file_path))
                    
            if self.worker.isRunning() and new_files:
                config = self.get_current_config()
                for file_path in new_files:
                    job = ProcessingJob(Path(file_path), config)
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
            
        config = self.get_current_config()
        self.jobs.clear()
        
        for i in range(self.file_list.count()):
            file_path = Path(self.file_list.item(i).text())
            job = ProcessingJob(file_path, config)
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
                
                output_dir = Path(self.get_current_config().output_dir) / f"output_{Path(file_path).stem}"
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
