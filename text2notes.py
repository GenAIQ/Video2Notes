# src/text2notes.py
"""
Module for generating structured notes from lecture transcriptions using Claude.
"""

import os
import anthropic
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

class NotesGenerator:
    """Class for generating structured notes from lecture transcriptions using Claude."""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize the notes generator with specified model."""
        load_dotenv()
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.model = model
        
    def _create_system_prompt(self, transcription: str) -> str:
        """Create the system prompt with the lecture transcription."""
        return f"""
You are an AI assistant tasked with creating detailed and enriched notes from a lecture transcription. Your role is to act as a top student in a computer science class who diligently takes notes and has a keen interest in including coding examples, mathematical and science prerequisites. Your goal is to transform the given lecture transcription into comprehensive, well-structured notes that will be useful for future reference and study.

Here is the lecture transcription you will be working with:
<lecture_transcription>
{transcription}
</lecture_transcription>

Create your notes following this structure:
1. Lecture Title
2. Pre-requisite Concepts Explanations
3. Detailed Notes
4. Python Examples

Follow these instructions for each section:

1. Lecture Title:
   - Extract or infer the main topic of the lecture from the transcription.
   - Present it as a clear, concise title.

2. Pre-requisite Concepts:
   - Explain key concepts that a student should understand before engaging with this lecture material.

3. Detailed Notes:
   - Organize the main content of the lecture into logical sections and subsections.
   - Use markdown formatting for headers (e.g., # for main sections, ## for subsections).
   - Include all important points, definitions, and explanations from the lecture.
   - If mathematical concepts are discussed, present them clearly using markdown's math formatting (e.g., $equation$).
   - Use bullet points or numbered lists where appropriate to enhance readability.
   - Include any examples or analogies provided in the lecture to illustrate concepts.

4. Python Examples:
   - Create all Python code examples that illustrate pre-requisite concepts and lecture notes from the lecture.
   - Ensure the examples are relevant, clear, and well-commented.
   - Use markdown code blocks to present the Python code (e.g., ```python).

General guidelines:
- Use markdown formatting throughout to enhance readability and structure.
- Be concise yet comprehensive in your note-taking.
- Ensure that your notes accurately reflect the content of the lecture without adding external information.
- If certain parts of the transcription are unclear, make a note of this in your detailed notes section.
"""

    def generate_notes(
        self,
        transcription: str,
        max_tokens: int = 8192,
        temperature: float = 0
    ) -> str:
        """
        Generate structured notes from a lecture transcription.
        
        Args:
            transcription: The lecture transcription text
            max_tokens: Maximum number of tokens in the response
            temperature: Temperature parameter for response generation
            
        Returns:
            Structured notes in markdown format
        """
        system_prompt = self._create_system_prompt(transcription)
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": "Please create detailed notes from the lecture transcription provided."
                }
            ]
        )
        
        return message.content[0].text


def save_notes(notes: str, output_path: Path) -> None:
    """Save the generated notes to a markdown file."""
    # Ensure the file has .md extension
    if not output_path.suffix == '.md':
        output_path = output_path.with_suffix('.md')
    output_path.write_text(notes)