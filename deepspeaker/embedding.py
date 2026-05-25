import os
import numpy as np
import torch
from deepspeaker.audio_ds import read_mfcc
from deepspeaker.batcher import sample_from_mfcc
from deepspeaker.constants import SAMPLE_RATE, NUM_FRAMES, WIN_LENGTH
from deepspeaker.conv_models import DeepSpeakerModel, load_weights_from_h5

def build_model(ckpt_path):
    model = DeepSpeakerModel()
    if os.path.exists(ckpt_path):
        try:
            load_weights_from_h5(model, ckpt_path)
        except Exception as e:
            print(f"Warning: Failed to load legacy Keras weights: {e}. Utilizing uninitialized weights.")
    else:
        print(f"Warning: DeepSpeaker checkpoint not found at {ckpt_path}. Model initialized with random weights.")
    model.eval()
    return model

def predict_embedding(model, audio, sr=SAMPLE_RATE, win_length=WIN_LENGTH, cuda=True):
    mfcc = sample_from_mfcc(read_mfcc(audio, sr, win_length), NUM_FRAMES)
    device = torch.device('cuda' if (cuda and torch.cuda.is_available()) else 'cpu')
    model = model.to(device)
    
    with torch.no_grad():
        x = torch.from_numpy(np.expand_dims(mfcc, axis=0)).float().to(device)
        embedding = model(x)
        embedding = embedding.cpu().numpy()
        
    return embedding
