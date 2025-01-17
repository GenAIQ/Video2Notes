# src/pipeline.py
"""
Main pipeline for converting video files to structured notes.
"""

import logging
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.logging import RichHandler

from videofile import Mp4VideoFinder, VideoFile
from video2audio import AudioConverterService, Mp3Converter
from audio2text import TranscriptionService, WhisperTranscriber
from text2notes import NotesGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("video2notes")
console = Console()


class VideoToNotesConverter:
    """Pipeline for converting video files to structured notes."""
    
    def __init__(
        self,
        whisper_model: str = "base",
        device: Optional[str] = None,
        claude_model: str = "claude-3-5-sonnet-20241022"
    ):
        """
        Initialize the conversion pipeline.
        
        Args:
            whisper_model: Name of the Whisper model to use
            device: Device to run Whisper on ("cuda" or "cpu")
            claude_model: Claude model to use for notes generation
        """
        # Initialize components
        self.video_finder = Mp4VideoFinder()
        self.audio_converter = AudioConverterService(Mp3Converter())
        self.transcriber = TranscriptionService(
            WhisperTranscriber(model_name=whisper_model, device=device)
        )
        self.notes_generator = NotesGenerator(model=claude_model)

    def process_single_video(
        self,
        video_path: Path,
        output_dir: Path
    ) -> Path:
        """
        Process a single video file through the complete pipeline.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory for output files
            
        Returns:
            Path to the generated notes file
        """
        logger.info(f"Processing video: {video_path}")
        
        try:
            # Create video file object
            video = VideoFile(
                filename=video_path.name,
                directory=str(video_path.parent),
                full_path=str(video_path),
                size_mb=video_path.stat().st_size / (1024 * 1024),
                modified_date=video_path.stat().st_mtime
            )
            
            # Create intermediary directories
            audio_dir = output_dir / "audio"
            transcription_dir = output_dir / "transcriptions"
            notes_dir = output_dir / "notes"
            
            for dir_path in [audio_dir, transcription_dir, notes_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Convert video to audio
            audio_path = audio_dir / f"{video_path.stem}.mp3"
            self.audio_converter.batch_convert([video], str(audio_dir))
            logger.info(f"Created audio file: {audio_path}")
            
            # Step 2: Transcribe audio
            transcription_result = self.transcriber.batch_transcribe(
                [str(audio_path)],
                output_dir=str(transcription_dir)
            )[0]
            logger.info(f"Created transcription for: {video_path.name}")
            
            # Step 3: Generate notes
            notes = self.notes_generator.generate_notes(transcription_result.text)
            notes_path = notes_dir / f"{video_path.stem}_notes.md"
            notes_path.write_text(notes)
            logger.info(f"Generated notes: {notes_path}")
            
            return notes_path
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")
            raise

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> List[Path]:
        """
        Process all videos in a directory through the pipeline.
        
        Args:
            input_dir: Directory containing video files
            output_dir: Directory for output files
            
        Returns:
            List of paths to generated notes files
        """
        logger.info(f"Processing directory: {input_dir}")
        
        try:
            # Find all videos
            videos = self.video_finder.find_videos(str(input_dir))
            
            if not videos:
                logger.warning(f"No MP4 videos found in {input_dir}")
                return []
                
            logger.info(f"Found {len(videos)} videos to process")
            
            # Process each video
            notes_paths = []
            for video in videos:
                video_path = Path(video.full_path)
                notes_path = self.process_single_video(video_path, output_dir)
                notes_paths.append(notes_path)
                
            return notes_paths
            
        except Exception as e:
            logger.error(f"Error processing directory {input_dir}: {str(e)}")
            raise


@click.command()
@click.argument(
    'input_path',
    type=click.Path(exists=True, path_type=Path)
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    help='Output directory for generated files'
)
@click.option(
    '--whisper-model', '-w',
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
    default='base',
    help='Whisper model to use for transcription'
)
@click.option(
    '--device', '-d',
    type=click.Choice(['cuda', 'cpu']),
    help='Device to run Whisper on (default: auto-detect)'
)
@click.option(
    '--claude-model', '-c',
    type=str,
    default="claude-3-5-sonnet-20241022",
    help='Claude model to use for notes generation'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose logging'
)
def main(
    input_path: Path,
    output_dir: Optional[Path],
    whisper_model: str,
    device: Optional[str],
    claude_model: str,
    verbose: bool
) -> None:
    """
    Convert video(s) to structured notes.
    
    This program takes either a single video file or a directory containing videos,
    converts them to audio, transcribes the audio, and generates structured notes
    from the transcription.
    
    INPUT_PATH can be either a single video file or a directory containing videos.
    """
    # Set logging level
    if verbose:
        logger.setLevel(logging.DEBUG)
        
    # Set up output directory
    if output_dir is None:
        output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize pipeline
        with console.status("[bold green]Initializing pipeline..."):
            pipeline = VideoToNotesConverter(
                whisper_model=whisper_model,
                device=device,
                claude_model=claude_model
            )
        
        # Process based on input type
        if input_path.is_file():
            if not input_path.suffix.lower() == '.mp4':
                raise click.BadParameter("Input file must be an MP4 video")
            pipeline.process_single_video(input_path, output_dir)
        else:
            pipeline.process_directory(input_path, output_dir)
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise click.Abort()


if __name__ == '__main__':
    main()