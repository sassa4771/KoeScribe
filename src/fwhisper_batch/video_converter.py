import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple


class VideoConverter:
    
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm', '.m4v', '.mpg', '.mpeg'}
    SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
    
    @staticmethod
    def is_ffmpeg_available() -> bool:
        return shutil.which('ffmpeg') is not None
    
    @staticmethod
    def is_video_file(file_path: Path) -> bool:
        return file_path.suffix.lower() in VideoConverter.SUPPORTED_VIDEO_FORMATS
    
    @staticmethod
    def is_audio_file(file_path: Path) -> bool:
        return file_path.suffix.lower() in VideoConverter.SUPPORTED_AUDIO_FORMATS
    
    @staticmethod
    def is_supported_file(file_path: Path) -> bool:
        return VideoConverter.is_video_file(file_path) or VideoConverter.is_audio_file(file_path)
    
    @staticmethod
    def convert_to_wav(input_path: Path, output_dir: Optional[Path] = None, 
                      sample_rate: int = 16000, channels: int = 1) -> Tuple[Path, bool]:
        if not VideoConverter.is_ffmpeg_available():
            raise RuntimeError("ffmpeg is not installed or not found in PATH")
        
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        is_video = VideoConverter.is_video_file(input_path)
        
        if output_dir is None:
            output_dir = input_path.parent / "converted_audio"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"{input_path.stem}_converted.wav"
        output_path = output_dir / output_filename
        
        if output_path.exists():
            return output_path, is_video
        
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-ar', str(sample_rate),
            '-ac', str(channels),
            '-vn',
            '-acodec', 'pcm_s16le',
            '-y',
            str(output_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            return output_path, is_video
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg conversion failed: {e.stderr}")
    
    @staticmethod
    def prepare_file_for_transcription(file_path: Path, temp_dir: Optional[Path] = None) -> Tuple[Path, bool]:
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.wav':
            return file_path, False
        
        if VideoConverter.is_video_file(file_path) or VideoConverter.is_audio_file(file_path):
            return VideoConverter.convert_to_wav(file_path, temp_dir)
        
        return file_path, False
