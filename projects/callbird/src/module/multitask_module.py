from dataclasses import asdict
from typing import Type, Optional
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer
from functools import partial
import torch
from torchmetrics import Accuracy

from birdset.modules.base_module import BaseModule
from birdset.configs import (
    NetworkConfig,
    LRSchedulerConfig,
    MultilabelMetricsConfig,
    LoggingParamsConfig,
)

class MultiTaskModule(BaseModule):
    def __init__(
        self,
        network: NetworkConfig = NetworkConfig(),
        loss: _Loss = BCEWithLogitsLoss(),
        optimizer: partial[Type[Optimizer]] = partial(AdamW, lr=1e-5),
        lr_scheduler: Optional[LRSchedulerConfig] = LRSchedulerConfig(),
        metrics_ebird: MultilabelMetricsConfig = MultilabelMetricsConfig(),
        metrics_calltype: MultilabelMetricsConfig = MultilabelMetricsConfig(),
        metrics_combined: MultilabelMetricsConfig = MultilabelMetricsConfig(),
        logging_params: LoggingParamsConfig = LoggingParamsConfig(),
        **kwargs,
    ):
        # Remove the unexpected argument before calling the parent constructor
        kwargs.pop("prediction_table", None)
        # We are handling metrics ourselves, so remove it before calling super
        kwargs.pop("metrics", None)

        super().__init__(
            network=network,
            loss=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=None, # Explicitly pass None for metrics
            **kwargs,
        )
        self.logging_params = logging_params

        # Metrics for ebird_code task
        self.train_metric_ebird = metrics_ebird.main_metric.clone()
        self.val_metric_ebird = metrics_ebird.main_metric.clone()
        self.test_metric_ebird = metrics_ebird.main_metric.clone()
        self.val_metric_best_ebird = metrics_ebird.val_metric_best.clone()

        # Metrics for call_type task
        self.train_metric_calltype = metrics_calltype.main_metric.clone()
        self.val_metric_calltype = metrics_calltype.main_metric.clone()
        self.test_metric_calltype = metrics_calltype.main_metric.clone()
        self.val_metric_best_calltype = metrics_calltype.val_metric_best.clone()

        # Metrics for combined task
        self.train_metric_combined = metrics_combined.main_metric.clone()
        self.val_metric_combined = metrics_combined.main_metric.clone()
        self.test_metric_combined = metrics_combined.main_metric.clone()
        self.val_metric_best_combined = metrics_combined.val_metric_best.clone()

    def model_step(self, batch, batch_idx):
        input_values, labels_ebird, labels_calltype = batch["input_values"], batch["labels_ebird"], batch["labels_calltype"]
        
        outputs = self.forward(input_values=input_values)
        logits_ebird = outputs["ebird_code"]
        logits_calltype = outputs["call_type"]

        loss_ebird = self.loss(logits_ebird, labels_ebird)
        loss_calltype = self.loss(logits_calltype, labels_calltype)
        
        total_loss = loss_ebird + loss_calltype

        return total_loss, logits_ebird, labels_ebird, logits_calltype, labels_calltype

    def training_step(self, batch, batch_idx):
        loss, logits_ebird, targets_ebird, logits_calltype, targets_calltype = self.model_step(batch, batch_idx)

        self.log(f"train/{self.loss.__class__.__name__}", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_metric_ebird(logits_ebird, targets_ebird.int())
        self.train_metric_calltype(logits_calltype, targets_calltype.int())
        self.train_metric_combined(torch.cat([logits_ebird, logits_calltype], dim=1), torch.cat([targets_ebird, targets_calltype], dim=1).int())

        self.log(f"train/ebird_{self.train_metric_ebird.__class__.__name__}", self.train_metric_ebird, **asdict(self.logging_params))
        self.log(f"train/calltype_{self.train_metric_calltype.__class__.__name__}", self.train_metric_calltype, **asdict(self.logging_params))
        self.log(f"train/combined_cmAP_{self.train_metric_combined.__class__.__name__}", self.train_metric_combined, **asdict(self.logging_params))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, logits_ebird, targets_ebird, logits_calltype, targets_calltype = self.model_step(batch, batch_idx)

        self.log(f"val/{self.loss.__class__.__name__}", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_metric_ebird(logits_ebird, targets_ebird.int())
        self.val_metric_calltype(logits_calltype, targets_calltype.int())
        self.val_metric_combined(torch.cat([logits_ebird, logits_calltype], dim=1), torch.cat([targets_ebird, targets_calltype], dim=1).int())

        self.log(f"val/ebird_{self.val_metric_ebird.__class__.__name__}", self.val_metric_ebird, **asdict(self.logging_params))
        self.log(f"val/calltype_{self.val_metric_calltype.__class__.__name__}", self.val_metric_calltype, **asdict(self.logging_params))
        self.log(f"val/combined_cmAP_{self.val_metric_combined.__class__.__name__}", self.val_metric_combined, **asdict(self.logging_params))

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        loss, logits_ebird, targets_ebird, logits_calltype, targets_calltype = self.model_step(batch, batch_idx)

        self.log(f"test/{self.loss.__class__.__name__}", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_metric_ebird(logits_ebird, targets_ebird.int())
        self.test_metric_calltype(logits_calltype, targets_calltype.int())
        self.test_metric_combined(torch.cat([logits_ebird, logits_calltype], dim=1), torch.cat([targets_ebird, targets_calltype], dim=1).int())

        self.log(f"test/ebird_{self.test_metric_ebird.__class__.__name__}", self.test_metric_ebird, **asdict(self.logging_params))
        self.log(f"test/calltype_{self.test_metric_calltype.__class__.__name__}", self.test_metric_calltype, **asdict(self.logging_params))
        self.log(f"test/combined_cmAP_{self.test_metric_combined.__class__.__name__}", self.test_metric_combined, **asdict(self.logging_params))

        return {"loss": loss}

    def on_train_start(self):
        """
        Resets the best validation metrics at the beginning of training.
        """
        self.val_metric_best_ebird.reset()
        self.val_metric_best_calltype.reset()
        self.val_metric_best_combined.reset()

    def on_validation_epoch_end(self):
        """
        Computes and logs the best validation metric at the end of the validation epoch.
        """
        # Ebird task
        val_metric_ebird = self.val_metric_ebird.compute()
        self.val_metric_best_ebird.update(val_metric_ebird)
        self.log(
            f"val/ebird_{self.val_metric_ebird.__class__.__name__}_best",
            self.val_metric_best_ebird.compute(),
            prog_bar=True
        )

        # Calltype task
        val_metric_calltype = self.val_metric_calltype.compute()
        self.val_metric_best_calltype.update(val_metric_calltype)
        self.log(
            f"val/calltype_{self.val_metric_calltype.__class__.__name__}_best",
            self.val_metric_best_calltype.compute(),
            prog_bar=True
        )

        # Combined task
        val_metric_combined = self.val_metric_combined.compute()
        self.val_metric_best_combined.update(val_metric_combined)
        self.log(
            f"val/combined_cmAP_{self.val_metric_combined.__class__.__name__}_best",
            self.val_metric_best_combined.compute(),
            prog_bar=True
        )