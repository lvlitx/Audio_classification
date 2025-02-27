from fastai.vision.all import *

def get_dataloaders(spectrogram_dir, batch_size=16, image_size=224):
    """
    Create DataLoaders for spectrogram images.
    
    Args:
        spectrogram_dir (str): Directory containing spectrogram images.
        batch_size (int): Batch size for training.
        image_size (int): Size to resize images.
    
    Returns:
        DataLoaders: FastAI DataLoaders object.
    """
    audio_data = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=GrandparentSplitter(train_name='train', valid_name='valid'),
        get_y=parent_label,
        item_tfms=[Resize(image_size)]
    )
    return audio_data.dataloaders(spectrogram_dir, bs=batch_size)

def train_model(spectrogram_dir, epochs=5, arch=resnet18):
    """
    Train a model on spectrogram images.
    
    Args:
        spectrogram_dir (str): Directory containing spectrogram images.
        epochs (int): Number of training epochs.
        arch: Model architecture (e.g., resnet18).
    
    Returns:
        Learner: FastAI Learner object.
    """
    dls = get_dataloaders(spectrogram_dir)
    learn = vision_learner(dls, arch, metrics=accuracy)
    learn.fine_tune(epochs)
    return learn