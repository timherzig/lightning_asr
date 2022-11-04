import torch

import pytorch_lightning as pl

from torch.utils.data import DataLoader

# Local imports 
from data.batch_transform import pad_batch
from data.commonvoice_dataset import CommonVoiceDataset

from models.emformer_model import Emformer_Model
from models.wav2vec2_model import Wav2Vec2_Model

def train(config, debug: False):
    print(f'Training: {config["model_name"]}\nDebug-mode: {"off" if not debug else "on"}')

    if config['model_type'] == 'emformer': model = Emformer_Model(config['emformer'], config['training']['bs'], config['training']['lr'])
    elif config['model_type'] == 'wav2vec2': model == Emformer_Model(config['wav2vec2'], config['training']['bs'], config['training']['lr'])

    if 'commonvoice' in config['data']['datasets'].split(' '):
        train_dataset = CommonVoiceDataset('train', config['data']['cv_loc'])
        train_dataloader = DataLoader(train_dataset, batch_size=config['training']['bs'], num_workers=config['training']['nw'], collate_fn=pad_batch)

        val_dataset = CommonVoiceDataset('dev', config['data']['cv_loc'])
        val_dataloader = DataLoader(val_dataset, batch_size=config['training']['bs'], num_workers=config['training']['nw'], collate_fn=pad_batch)

    trainer = pl.Trainer(accelerator='gpu', devices=[1], max_epochs=(1 if debug else 1000))

    trainer.fit(model, train_dataloader, val_dataloader)



