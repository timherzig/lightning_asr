import torch

import pytorch_lightning as pl

from data.commonvoice_dataset import CommonVoiceDataset
from models.emformer_model import Emformer_Model
from models.wav2vec2_model import Wav2Vec2_Model

def train(config, debug: False):
    print(f'Training: {config["model_name"]}\nDebug-mode: {"off" if not debug else "on"}')

    if config['model_type'] == 'emformer': model = Emformer_Model(config['emformer'], config['training']['bs'], config['training']['lr'])
    elif config['model_type'] == 'wav2vec2': model == Emformer_Model(config['wav2vec2'], config['training']['bs'], config['training']['lr'])

    print(model)

