import torch
from nni.nas.evaluator.pytorch import ClassificationModule
from torch.nn import DataParallel


class DartsClassificationModule(ClassificationModule):
    def __init__(
        self,
        learning_rate: float = 0.001,
        weight_decay: float = 0.,
        auxiliary_loss_weight: float = 0.4,
        max_epochs: int = 600,
        num_classes: int = 10,
        lr_final: float = 1e-3
    ):
        self.auxiliary_loss_weight = auxiliary_loss_weight
        # Training length will be used in LR scheduler
        self.max_epochs = max_epochs
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay, export_onnx=False, num_classes=num_classes)
        
    def configure_optimizers(self):
        """Customized optimizer with momentum, as well as a scheduler."""
        optimizer = torch.optim.SGD(
            self.parameters(),
            momentum=0.9,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        # Cosine annealing scheduler with T_max equal to total epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=1e-3
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def training_step(self, batch, batch_idx):
        """Training step, customized with auxiliary loss and flexible unpacking."""
        x, y = batch
        out = self(x)
        # Handle auxiliary output if present
        if self.auxiliary_loss_weight and isinstance(out, (tuple, list)) and len(out) == 2:
            y_hat, y_aux = out
            loss_main = self.criterion(y_hat, y)
            loss_aux = self.criterion(y_aux, y)
            self.log('train_loss_main', loss_main)
            self.log('train_loss_aux', loss_aux)
            loss = loss_main + self.auxiliary_loss_weight * loss_aux
        else:
            # single output or no auxiliary
            y_hat = out[0] if isinstance(out, (tuple, list)) else out
            loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        for name, metric in self.metrics.items():
            self.log('train_' + name, metric(y_hat, y), prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        # Handle DataParallel wrapper when adjusting drop path
        model = self.trainer.model
        if isinstance(model, DataParallel):
            target_model = model.module
        else:
            target_model = model

        # Set drop path probability before every epoch, scaled by epoch ratio
        if hasattr(target_model, 'set_drop_path_prob') and hasattr(target_model, 'drop_path_prob'):
            drop_prob = target_model.drop_path_prob * self.current_epoch / self.max_epochs
            target_model.set_drop_path_prob(drop_prob)

        # Logging learning rate at the beginning of every epoch
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr)
