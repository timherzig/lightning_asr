import torch
import pytorch_lightning as pl

class Wav2Vec2_Model(pl.LightningModule):

    def __init__(self, model_args, bs, lr) -> None:
        super().__init__()

        self.bs = bs
        self.lr = lr