# Video2Notes

A comprehensive pipeline for converting video lectures to structured markdown notes. This tool automates the process of:
1. Converting video files to audio
2. Transcribing audio to text using Whisper
3. Generating structured markdown notes using Claude

## Features

- **Video Processing**:
  - Handles both single videos and directories
  - Supports MP4 video format
  - Organizes outputs in structured directories

- **Audio Conversion**:
  - Converts videos to MP3 format
  - Uses moviepy for efficient conversion
  - Maintains audio quality

- **Transcription**:
  - Uses OpenAI's Whisper for accurate transcription
  - Supports multiple languages
  - Configurable model sizes (tiny to large)
  - GPU acceleration support

- **Notes Generation**:
  - Uses Anthropic's Claude for intelligent note generation
  - Produces well-structured markdown notes
  - Includes code examples and prerequisites
  - Maintains consistent formatting

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd Video2Notes
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Create a `.env` file in the project root
- Add your Anthropic API key:
```bash
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

Process a single video:
```bash
python main.py video.mp4 -o output_dir
```
Process a directory of videos:
```bash
python main.py videos_directory -o output_dir
```

### Advanced Options

```bash
python main.py input_path \
    --output-dir output_dir \
    --whisper-model medium \
    --device cuda \
    --claude-model claude-3-5-sonnet-20241022 \
    --verbose
```

Options:
- `--whisper-model`: Choose Whisper model size (tiny/base/small/medium/large)
- `--device`: Select processing device (cuda/cpu)
- `--claude-model`: Specify Claude model version
- `--verbose`: Enable detailed logging
- `-o, --output-dir`: Specify output directory

## Output Structure

```
output_dir/
├── audio/              # MP3 files converted from videos
├── transcriptions/     # Raw transcription text files
└── notes/             # Final markdown notes
```

## Project Structure

```
src/
├── __init__.py
├── audio2text.py      # Audio transcription using Whisper
├── audio2video.py     # Video to audio conversion
├── main.py        # Main pipeline orchestration
├── text2notes.py      # Notes generation using Claude
└── videofile.py       # Video file handling utilities
```

## Dependencies

- `torch` and `whisper`: For audio transcription
- `moviepy`: For video processing
- `anthropic`: For Claude API access
- `click`: For command-line interface
- `rich`: For enhanced console output
- `python-dotenv`: For environment variable management
- `pandas`: For data handling

## Environment Setup

The project requires:
- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster processing)
- FFmpeg (for video processing)

## Error Handling

The pipeline includes comprehensive error handling:
- Validates input files and formats
- Checks for required API keys
- Provides detailed error messages
- Maintains logging throughout the process

## License

MIT 

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Contact

[Your contact information]
