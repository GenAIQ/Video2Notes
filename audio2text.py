# src/audio2text.py
"""
Module for transcribing audio files to text using Whisper.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import whisper


@dataclass(frozen=True)
class TranscriptionResult:
    """Immutable data class representing the result of an audio transcription."""

    text: str
    audio_path: str
    language: str
    segments: List[Dict]
    model_used: str

    def __post_init__(self) -> None:
        """Validate the transcription result data after initialization."""
        if not self.text:
            raise ValueError("Transcription text cannot be empty")
        if not self.audio_path:
            raise ValueError("Audio path cannot be empty")
        if not self.language:
            raise ValueError("Language cannot be empty")


class AudioTranscriber(ABC):
    """Abstract base class for audio transcription operations."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to the input audio file.

        Returns:
            TranscriptionResult containing the transcription and metadata.

        Raises:
            FileNotFoundError: If the input audio file doesn't exist.
            RuntimeError: If there are issues during transcription.
        """
        pass


class WhisperTranscriber(AudioTranscriber):
    """Implementation of AudioTranscriber using OpenAI's Whisper model."""

    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        """
        Initialize the Whisper transcriber.

        Args:
            model_name: Name of the Whisper model to use (e.g., "base", "small", "medium").
            device: Device to run the model on ("cuda" or "cpu"). If None, automatically
                   selects CUDA if available, else CPU.
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()

    def _load_model(self) -> whisper.Whisper:
        """Load the Whisper model."""
        try:
            return whisper.load_model(self.model_name).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {str(e)}")

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe an audio file using the Whisper model."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            result = self.model.transcribe(str(audio_path))
            return TranscriptionResult(
                text=result["text"],
                audio_path=str(audio_path),
                language=result.get("language", "unknown"),
                segments=result.get("segments", []),
                model_used=self.model_name
            )
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}")


class TranscriptionService:
    """Service class for handling batch audio transcription operations."""

    def __init__(self, transcriber: AudioTranscriber):
        """Initialize with specific transcriber implementation."""
        self.transcriber = transcriber

    def batch_transcribe(
        self,
        audio_files: List[str],
        output_dir: Optional[str] = None
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio files and optionally save the results.

        Args:
            audio_files: List of paths to audio files to transcribe.
            output_dir: Optional directory to save transcription results.

        Returns:
            List of TranscriptionResult objects.
        """
        results = []
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        for audio_file in audio_files:
            result = self.transcriber.transcribe(audio_file)
            results.append(result)

            if output_dir:
                self._save_transcription(result, output_path)

        return results

    def _save_transcription(self, result: TranscriptionResult, output_dir: Path) -> None:
        """
        Save a transcription result to a text file.

        Args:
            result: TranscriptionResult to save.
            output_dir: Directory to save the transcription file.
        """
        audio_path = Path(result.audio_path)
        output_file = output_dir / f"{audio_path.stem}_transcription.txt"
        
        output_file.write_text(
            f"Transcription of {audio_path.name}\n"
            f"Language: {result.language}\n"
            f"Model: {result.model_used}\n\n"
            f"{result.text}\n"
        )