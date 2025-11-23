import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from nni.nas.evaluator.pytorch import ClassificationModule

class DartsClassificationModule(ClassificationModule):
    def __init__(
        self,
        learning_rate: float = 0.001,
        weight_decay: float = 0.,
        auxiliary_loss_weight: float = 0.4,
        max_epochs: int = 600,
        num_classes: int = 10,
        lr_final: float = 1e-5,
        warmup_epochs: int = 0,
        label_smoothing: float = 0.0,
        optimizer: str = "SGD",
    ):
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.max_epochs = max_epochs
        self.lr_final = lr_final
        self.warmup_epochs = max(0, int(warmup_epochs))
        self._drop_path_max = None
        self.my_optimizer = optimizer

        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            export_onnx=False,
            num_classes=num_classes
        )

        if label_smoothing and label_smoothing > 0.0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def configure_optimizers(self):
        if self.my_optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                momentum=0.9,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.my_optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.my_optimizer}")

        t_max = max(1, self.max_epochs - self.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=self.lr_final)

        if self.warmup_epochs > 0:
            warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=self.warmup_epochs)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs])
        else:
            scheduler = cosine

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)

        if self.auxiliary_loss_weight and isinstance(out, (tuple, list)) and len(out) == 2:
            y_hat, y_aux = out
            loss_main = self.criterion(y_hat, y)
            loss_aux = self.criterion(y_aux, y)
            loss = loss_main + self.auxiliary_loss_weight * loss_aux
            self.log('train_loss_main', loss_main, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
            self.log('train_loss_aux', loss_aux, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
        else:
            y_hat = out[0] if isinstance(out, (tuple, list)) else out
            loss = self.criterion(y_hat, y)

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        preds = y_hat.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log('train_acc', acc, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)

        if isinstance(out, (tuple, list)):
            y_hat = out[0]
        else:
            y_hat = out

        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        preds = y_hat.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log('val_acc', acc, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def on_train_epoch_start(self):
        try:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('lr', lr, prog_bar=True, on_epoch=True, on_step=False)
        except Exception:
            pass

        model = getattr(self, 'model', None)

        if model is not None:
            if self._drop_path_max is None:
                if hasattr(model, 'drop_path_prob'):
                    self._drop_path_max = float(getattr(model, 'drop_path_prob'))
                else:
                    self._drop_path_max = 0.0

            ratio = min(1.0, float(self.current_epoch + 1) / float(max(1, self.max_epochs)))
            drop_prob = self._drop_path_max * ratio

            if hasattr(model, 'set_drop_path_prob') and callable(getattr(model, 'set_drop_path_prob')):
                model.set_drop_path_prob(drop_prob)
            elif hasattr(model, 'drop_path_prob'):
                setattr(model, 'drop_path_prob', drop_prob)