import sys
import os
import json
import threading
import time
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
from PySide6.QtCore import QThread, Signal, QTimer, Qt, QSettings
from PySide6.QtGui import QFont, QIcon

import pandas as pd
from faster_whisper import WhisperModel

from .transcribe_batch import (
    transcribe_one, load_config, resolve_device, resolve_compute_type,
    detect_device, is_diarization_enabled
)


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
    max_speakers: int = 10
    output_dir: str = "./outputs"


@dataclass
class ProcessingJob:
    file_path: Path
    project_config: ProjectConfig
    status: str = "待機中"
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TranscriptionWorker(QThread):
    progress_updated = Signal(str, float, str)  # file_path, progress, stage
    job_completed = Signal(str, dict)  # file_path, result
    job_failed = Signal(str, str)  # file_path, error
    
    def __init__(self):
        super().__init__()
        self.jobs: List[ProcessingJob] = []
        self.current_job: Optional[ProcessingJob] = None
        self.model: Optional[WhisperModel] = None
        self.should_stop = False
        
    def add_job(self, job: ProcessingJob):
        self.jobs.append(job)
        
    def stop_processing(self):
        self.should_stop = True
        
    def run(self):
        while self.jobs and not self.should_stop:
            job = self.jobs.pop(0)
            self.current_job = job
            
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
                
                self.progress_updated.emit(str(job.file_path), 100.0, "完了")
                self.job_completed.emit(str(job.file_path), result)
                
            except Exception as e:
                self.job_failed.emit(str(job.file_path), str(e))
                
        self.current_job = None
        
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
        
        self.setup_ui()
        self.setup_connections()
        self.load_last_project()
        
    def setup_ui(self):
        self.setWindowTitle("FWhisper Batch GUI - 音声文字起こし & 話者分離")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        
        project_group = QGroupBox("プロジェクト管理")
        project_layout = QVBoxLayout(project_group)
        
        project_select_layout = QHBoxLayout()
        self.project_combo = QComboBox()
        self.project_combo.setEditable(True)
        project_select_layout.addWidget(QLabel("プロジェクト:"))
        project_select_layout.addWidget(self.project_combo)
        
        project_buttons_layout = QHBoxLayout()
        self.save_project_btn = QPushButton("保存")
        self.load_project_btn = QPushButton("読み込み")
        self.delete_project_btn = QPushButton("削除")
        project_buttons_layout.addWidget(self.save_project_btn)
        project_buttons_layout.addWidget(self.load_project_btn)
        project_buttons_layout.addWidget(self.delete_project_btn)
        
        project_layout.addLayout(project_select_layout)
        project_layout.addLayout(project_buttons_layout)
        
        settings_group = QGroupBox("設定")
        settings_layout = QFormLayout(settings_group)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["large-v3", "large-v2", "large", "medium", "small", "base", "tiny"])
        settings_layout.addRow("モデル:", self.model_combo)
        
        self.language_combo = QComboBox()
        self.language_combo.addItems(["ja", "en", "auto"])
        settings_layout.addRow("言語:", self.language_combo)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        settings_layout.addRow("デバイス:", self.device_combo)
        
        self.beam_size_spin = QSpinBox()
        self.beam_size_spin.setRange(1, 10)
        self.beam_size_spin.setValue(5)
        settings_layout.addRow("ビームサイズ:", self.beam_size_spin)
        
        self.vad_check = QCheckBox("音声活動検出 (VAD)")
        self.vad_check.setChecked(True)
        settings_layout.addRow(self.vad_check)
        
        self.min_silence_spin = QSpinBox()
        self.min_silence_spin.setRange(100, 2000)
        self.min_silence_spin.setValue(500)
        self.min_silence_spin.setSuffix(" ms")
        settings_layout.addRow("最小無音時間:", self.min_silence_spin)
        
        self.diarization_check = QCheckBox("話者分離を有効化")
        self.diarization_check.setChecked(True)
        settings_layout.addRow(self.diarization_check)
        
        self.max_speakers_spin = QSpinBox()
        self.max_speakers_spin.setRange(2, 20)
        self.max_speakers_spin.setValue(10)
        settings_layout.addRow("最大話者数:", self.max_speakers_spin)
        
        self.output_dir_edit = QLineEdit("./outputs")
        self.output_dir_btn = QPushButton("参照")
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(self.output_dir_btn)
        settings_layout.addRow("出力フォルダ:", output_layout)
        
        left_layout.addWidget(project_group)
        left_layout.addWidget(settings_group)
        left_layout.addStretch()
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        file_group = QGroupBox("ファイル選択")
        file_layout = QVBoxLayout(file_group)
        
        file_buttons_layout = QHBoxLayout()
        self.add_files_btn = QPushButton("ファイル追加")
        self.add_folder_btn = QPushButton("フォルダ追加")
        self.clear_files_btn = QPushButton("クリア")
        file_buttons_layout.addWidget(self.add_files_btn)
        file_buttons_layout.addWidget(self.add_folder_btn)
        file_buttons_layout.addWidget(self.clear_files_btn)
        
        self.file_list = QListWidget()
        
        file_layout.addLayout(file_buttons_layout)
        file_layout.addWidget(self.file_list)
        
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("処理開始")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addStretch()
        
        progress_group = QGroupBox("処理状況")
        progress_layout = QVBoxLayout(progress_group)
        
        self.current_file_label = QLabel("待機中...")
        self.progress_bar = QProgressBar()
        self.stage_label = QLabel("")
        
        progress_layout.addWidget(self.current_file_label)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.stage_label)
        
        results_group = QGroupBox("結果")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["ファイル", "状態", "処理時間", "出力"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        
        results_layout.addWidget(self.results_table)
        
        right_layout.addWidget(file_group)
        right_layout.addLayout(control_layout)
        right_layout.addWidget(progress_group)
        right_layout.addWidget(results_group)
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
    def setup_connections(self):
        self.save_project_btn.clicked.connect(self.save_project)
        self.load_project_btn.clicked.connect(self.load_project)
        self.delete_project_btn.clicked.connect(self.delete_project)
        
        self.add_files_btn.clicked.connect(self.add_files)
        self.add_folder_btn.clicked.connect(self.add_folder)
        self.clear_files_btn.clicked.connect(self.clear_files)
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        
        self.start_btn.clicked.connect(self.start_processing)
        self.stop_btn.clicked.connect(self.stop_processing)
        
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.job_completed.connect(self.job_completed)
        self.worker.job_failed.connect(self.job_failed)
        
        self.refresh_project_list()
        
    def refresh_project_list(self):
        self.project_combo.clear()
        projects = self.project_manager.list_projects()
        self.project_combo.addItems(projects)
        
    def save_project(self):
        name = self.project_combo.currentText().strip()
        if not name:
            QMessageBox.warning(self, "警告", "プロジェクト名を入力してください。")
            return
            
        config = self.get_current_config()
        config.name = name
        self.project_manager.save_project(config)
        self.refresh_project_list()
        QMessageBox.information(self, "保存完了", f"プロジェクト '{name}' を保存しました。")
        
    def load_project(self):
        name = self.project_combo.currentText().strip()
        if not name:
            return
            
        config = self.project_manager.load_project(name)
        if config:
            self.apply_config(config)
            QMessageBox.information(self, "読み込み完了", f"プロジェクト '{name}' を読み込みました。")
        else:
            QMessageBox.warning(self, "エラー", f"プロジェクト '{name}' が見つかりません。")
            
    def delete_project(self):
        name = self.project_combo.currentText().strip()
        if not name:
            return
            
        reply = QMessageBox.question(
            self, "確認", 
            f"プロジェクト '{name}' を削除しますか？",
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
            
    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "フォルダを選択")
        if folder:
            folder_path = Path(folder)
            extensions = ['.wav', '.mp3', '.m4a', '.mp4', '.avi', '.mov', '.flv']
            
            for ext in extensions:
                for file_path in folder_path.glob(f"*{ext}"):
                    item = QListWidgetItem(str(file_path))
                    self.file_list.addItem(item)
                    
    def clear_files(self):
        self.file_list.clear()
        
    def select_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "出力フォルダを選択")
        if folder:
            self.output_dir_edit.setText(folder)
            
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
                
        if not self.worker.jobs and not self.worker.current_job:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.current_file_label.setText("すべての処理が完了しました")
            self.progress_bar.setValue(100)
            self.stage_label.setText("完了")
            
    def job_failed(self, file_path: str, error: str):
        file_name = Path(file_path).name
        
        for i in range(self.results_table.rowCount()):
            if self.results_table.item(i, 0).text() == file_name:
                self.results_table.setItem(i, 1, QTableWidgetItem("エラー"))
                self.results_table.setItem(i, 3, QTableWidgetItem(error))
                break
                
        QMessageBox.warning(self, "処理エラー", f"ファイル '{file_name}' の処理中にエラーが発生しました:\n{error}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FWhisper Batch GUI")
    app.setOrganizationName("FWhisper")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
