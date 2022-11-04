import torch
import pytorch_lightning as pl

from torchaudio.models import Emformer

class Emformer_Model(pl.LightningModule):

    def __init__(self, model_args, bs, lr) -> None:
        super().__init__()

        self.bs = bs
        self.lr = lr

        self.emformer = Emformer(
            input_dim=model_args['input_dim'],
            num_heads=model_args['num_heads'],
            ffn_dim=model_args['ffn_dim'],
            num_layers=model_args['num_layers'],
            segment_length=model_args['segment_length'],
            dropout=model_args['dropout'],
            activation=model_args['activation'],
            left_context_length=model_args['left_context_length'],
            right_context_length=model_args['right_context_length'],
            max_memory_size=model_args['max_memory_size'],
            weight_init_scale_strategy=model_args['weight_init_scale_strategy'],
            tanh_on_mem=model_args['tanh_on_mem'],
            negative_inf=model_args['negative_inf']
        )

    def forward(self, x):
        return self.emformer(x)

    def training_step(self, batch, batch_idx):
        x, x_lengths, y = batch
        o, o_lengths = self.emformer(x, x_lengths)
        ctc_loss = torch.nn.CTCLoss()
        loss = ctc_loss(o, y, o_lengths, len(y))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_lengths, y = batch
        o, o_lengths = self.emformer(x, x_lengths)
        ctc_loss = torch.nn.CTCLoss()
        loss = ctc_loss(o, y, o_lengths, len(y))
        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, x_lengths, y = batch
        o, o_lengths = self.emformer(x, x_lengths)
        ctc_loss = torch.nn.CTCLoss()
        loss = ctc_loss(o, y, o_lengths, len(y))
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer