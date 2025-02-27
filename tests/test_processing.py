import os
import pytest
from src.audio_processing import audio_to_spectrogram

def test_audio_to_spectrogram(tmp_path):
    # Create a dummy audio file (you can replace this with a real file)
    dummy_audio = tmp_path / "test.wav"
    dummy_audio.write_text("dummy data")  # Replace with actual audio data
    
    # Test spectrogram generation
    output_path = tmp_path / "test.png"
    audio_to_spectrogram(str(dummy_audio), str(output_path))
    
    # Check if the output file exists
    assert os.path.exists(output_path)