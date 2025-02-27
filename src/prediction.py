from fastai.vision.all import *
import tempfile

def predict_audio_quality(audio_path, learner):
    """
    Predict the quality of an audio file using a trained model.
    
    Args:
        audio_path (str): Path to the input audio file.
        learner: Trained FastAI Learner object.
    
    Returns:
        dict: Prediction and confidence.
    """
    with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
        audio_to_spectrogram(audio_path, tmp.name)
        pred, _, probs = learner.predict(tmp.name)
        return {'prediction': pred, 'confidence': float(probs.max())}