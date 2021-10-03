import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim, threshold
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from argparse import ArgumentParser
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

class LightningInterface(pl.LightningModule):
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__()
        self.best_f1 = 0.
        self.threshold = threshold
        self.criterion = nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_nb, optimizer_idx=0):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        # import pdb; pdb.set_trace()
        # self.log('lr', self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0], on_step=True)
        # self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True)
        return {'loss': loss, 'log': tensorboard_logs}

    # def training_epoch_end(self, output) -> None:
    #     self.log('lr', self.hparams.lr)
    
    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        yy, yy_hat = y.detach().cpu().numpy(), y_hat.sigmoid().detach().cpu().numpy()
        return {'val_loss': self.criterion(y_hat, y), "labels": yy, "probs": yy_hat}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        all_labels = np.concatenate([x['labels'] for x in outputs])
        all_probs = np.concatenate([x['probs'] for x in outputs])
        all_preds = (all_probs > self.threshold).astype(int)
        acc = np.mean(all_labels == all_preds)
        p = precision_score(all_labels, all_preds, average='micro')
        r = recall_score(all_labels, all_preds, average='micro')
        f1 = f1_score(all_labels, all_preds, average='micro')
        self.best_f1 = max(self.best_f1, f1)
        # return {'val_loss': avg_loss, 'val_acc': avg_acc, 'hp_metric': self.best_acc}
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc, 'val_p': p, 'val_r': r, 'val_f1': f1, 'hp_metric': self.best_f1}
        # import pdb; pdb.set_trace()
        self.log('best_f1', self.best_f1, prog_bar=True, on_epoch=True)
        self.log_dict(tensorboard_logs)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        yy, yy_hat = y.detach().cpu().numpy(), y_hat.sigmoid().detach().cpu().numpy()
        return {'test_loss': self.criterion(y_hat, y), "labels": yy, "probs": yy_hat}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        all_labels = np.concatenate([x['labels'] for x in outputs])
        all_probs = np.concatenate([x['probs'] for x in outputs])
        all_preds = (all_probs > self.threshold).astype(int)
        acc = np.mean(all_labels == all_preds)
        p = precision_score(all_labels, all_preds, average='micro')
        r = recall_score(all_labels, all_preds, average='micro')
        f1 = f1_score(all_labels, all_preds, average='micro')
        report = classification_report(all_labels, all_preds, target_names=["add_tail", "repeat_event", "repeat_word", "delete_word", "error"])
        print(report)
        results = {'test_loss': avg_loss, 'test_acc': acc, 'test_p': p, 'test_r': r, 'test_f1': f1}
        self.log_dict(results, on_epoch=True, prog_bar=True)
        return results

    def on_after_backward(self):
        pass
        # can check gradient
        # global_step = self.global_step
        # if int(global_step) % 100 == 0:
        #     for name, param in self.named_parameters():
        #         self.logger.experiment.add_histogram(name, param, global_step)
        #         if param.requires_grad:
        #             self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser


class Classifier(LightningInterface):
    def __init__(self, threshold=0.5, num_classes=5, lr=5e-5, model_type="prajjwal1/bert-tiny", **kwargs):
        super().__init__(**kwargs)

        self.model_type = model_type
        self.model = BERTFlatClassifier(model_type, num_classes=num_classes)
        self.threshold = threshold
        self.lr = lr
        # self.lr_sched = lr_sched
        self.save_hyperparameters()
        print(self.hparams)

    def forward(self, x):
        x = self.model(**x)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = LightningInterface.add_model_specific_args(parser)
        parser.add_argument("--threshold", type=float, default=0.5)
        parser.add_argument("--lr", type=float, default=2e-4)
        parser.add_argument("--num_classes", type=int, default=5)
        # parser.add_argument("--lr_sched", type=str, default="none")
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        # if self.lr_sched == "none":
        #     return optimizer
        # elif self.lr_sched == 'reduce':
        #     return {
        #             'optimizer': optimizer, 
        #             'lr_scheduler': {
        #                 'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2),
        #                 'monitor': 'val_acc'
        #             }
        #         }
        # else:
        #     schedulers = {
        #         'exp': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8),
        #         'cosine': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3),
        #         'cyclic': optim.lr_scheduler.CyclicLR(optimizer, 5e-5, 3e-4, 2, cycle_momentum=False)
        #     }
        #     return {'optimizer': optimizer, 'lr_scheduler': schedulers[self.lr_sched]}


class BERTFlatClassifier(nn.Module):
    def __init__(self, model_type, num_classes=5) -> None:
        super().__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        self.encoder = AutoModel.from_pretrained(model_type)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.encoder.config.hidden_size, num_classes)
    
    def forward(self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs):
        outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        x = outputs.last_hidden_state[:, 0, :]
        # x = outputs.last_hidden_state.mean(1)  # [bs, seq_len, hidden_size] -> [bs, hidden_size]
        x = self.dropout(x)
        logits = self.clf(x)
        return logits

