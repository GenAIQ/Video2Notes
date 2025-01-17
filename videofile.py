# src/videofile.py
"""
Module for handling video file operations and discovery.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import os
import pandas as pd
from typing import List, Dict


@dataclass
class VideoFile:
    """Data class representing a video file with its metadata."""
    filename: str
    directory: str
    full_path: str
    size_mb: float
    modified_date: datetime


class VideoFinder(ABC):
    """Abstract base class for video file discovery."""
    
    @abstractmethod
    def find_videos(self, root_dir: str) -> List[VideoFile]:
        """
        Find video files in the specified directory.
        
        Args:
            root_dir: Root directory to search for videos.
            
        Returns:
            List of VideoFile objects found.
        """
        pass


class Mp4VideoFinder(VideoFinder):
    """Implementation of VideoFinder for MP4 files."""
    
    def find_videos(self, root_dir: str) -> List[VideoFile]:
        """Find all MP4 files in the specified directory."""
        videos = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.mp4'):
                    videos.append(self._create_video_file(root, file))
        return videos

    def _create_video_file(self, root: str, filename: str) -> VideoFile:
        """Create a VideoFile object from a file path."""
        full_path = os.path.join(root, filename)
        stats = os.stat(full_path)
        return VideoFile(
            filename=filename,
            directory=root,
            full_path=full_path,
            size_mb=stats.st_size / (1024 * 1024),
            modified_date=datetime.fromtimestamp(stats.st_mtime)
        )


class DataFrameConverter:
    """Utility class for converting video files to pandas DataFrames."""
    
    def convert(self, videos: List[VideoFile]) -> pd.DataFrame:
        """Convert a list of VideoFile objects to a DataFrame."""
        return pd.DataFrame([self._to_dict(video) for video in videos])

    def _to_dict(self, video: VideoFile) -> Dict:
        """Convert a single VideoFile to a dictionary."""
        return {
            'filename': video.filename,
            'directory': video.directory,
            'full_path': video.full_path,
            'size_mb': video.size_mb,
            'modified_date': video.modified_date
        }