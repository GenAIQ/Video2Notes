# src/audio2video.py
"""
Module for converting video files to audio format.
"""

from abc import ABC, abstractmethod
from moviepy import VideoFileClip
from pathlib import Path
from typing import List

from videofile import VideoFile


class AudioConverter(ABC):
    """Abstract base class for audio conversion operations."""
    
    @abstractmethod
    def convert(self, video_path: str, output_path: str) -> None:
        """
        Convert a video file to audio format.
        
        Args:
            video_path: Path to the input video file.
            output_path: Path where the audio file should be saved.
        """
        pass


class Mp3Converter(AudioConverter):
    """Implementation of AudioConverter for MP3 format."""
    
    def convert(self, video_path: str, output_path: str) -> None:
        """Convert a video file to MP3 format."""
        with VideoFileClip(video_path) as clip:
            clip.audio.write_audiofile(output_path)


class AudioConverterService:
    """Service class for handling batch audio conversion operations."""
    
    def __init__(self, converter: AudioConverter):
        """Initialize with specific converter implementation."""
        self.converter = converter
    
    def batch_convert(self, videos: List[VideoFile], output_dir: str) -> None:
        """
        Convert multiple videos to audio format.
        
        Args:
            videos: List of VideoFile objects to convert.
            output_dir: Directory where audio files should be saved.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for video in videos:
            audio_path = output_path / f"{Path(video.filename).stem}.mp3"
            self.converter.convert(video.full_path, str(audio_path))