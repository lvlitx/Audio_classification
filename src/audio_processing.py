import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def audio_to_spectrogram(audio_path, save_path, sr=16000, n_mels=128):
    """
    Convert an audio file to a spectrogram and save it as an image.
    
    Args:
        audio_path (str): Path to the input audio file.
        save_path (str): Path to save the spectrogram image.
        sr (int): Sample rate for audio loading.
        n_mels (int): Number of mel bands for the spectrogram.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_spectrograms(input_dir, output_dir):
    """
    Generate spectrograms for all audio files in a directory.
    
    Args:
        input_dir (str): Directory containing audio files.
        output_dir (str): Directory to save spectrogram images.
    """
    os.makedirs(output_dir, exist_ok=True)
    for label in ['good', 'bad']:
        label_path = os.path.join(input_dir, label)
        save_label_path = os.path.join(output_dir, label)
        os.makedirs(save_label_path, exist_ok=True)
        for audio_file in os.listdir(label_path):
            if audio_file.endswith('.wav'):
                audio_path = os.path.join(label_path, audio_file)
                save_path = os.path.join(save_label_path, f"{os.path.splitext(audio_file)[0]}.png")
                audio_to_spectrogram(audio_path, save_path)