from dataclasses import asdict
from typing import Type, Optional
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer
from functools import partial

from torchmetrics import MetricCollection

from birdset.modules.base_module import BaseModule
from birdset.configs import (
    NetworkConfig,
    LRSchedulerConfig,
    MultilabelMetricsConfig,
    LoggingParamsConfig,
)
from birdset.modules.metrics.multilabel import cmAP
from torchmetrics.classification import MultilabelAccuracy, MultilabelExactMatch

class NaiveMultiTaskModule(BaseModule):
    def __init__(
        self,
        network: NetworkConfig = NetworkConfig(),
        loss: _Loss = BCEWithLogitsLoss(),
        optimizer: partial[Type[Optimizer]] = partial(AdamW, lr=1e-5),
        lr_scheduler: Optional[LRSchedulerConfig] = LRSchedulerConfig(),
        metrics_ebird: MultilabelMetricsConfig = MultilabelMetricsConfig(),
        metrics_calltype: None = None,
                metrics_combined: MetricCollection = MetricCollection(
            [
                cmAP(num_labels=106),
                MultilabelAccuracy(num_labels=106),
                MultilabelExactMatch(num_labels=106),
            ]
        ),
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

        # Metrics for combined task
        self.train_metric_combined = metrics_combined.clone()
        self.val_metric_combined = metrics_combined.clone()
        self.test_metric_combined = metrics_combined.clone()
        self.val_metric_best_combined_exact = (
            metrics_ebird.val_metric_best.clone()
        )  # Use one of the val_metric_best
        self.val_metric_best_combined_cmap = (
            metrics_ebird.val_metric_best.clone()
        )  # Use one of the val_metric_best

    def model_step(self, batch, batch_idx):
        input_values, labels_ebird = batch["input_values"], batch["labels_ebird"]

        outputs = self.forward(input_values=input_values)
        logits_ebird = outputs["ebird_code"]

        loss_ebird = self.loss(logits_ebird, labels_ebird)
        
        total_loss = loss_ebird

        return total_loss, logits_ebird, labels_ebird

    def training_step(self, batch, batch_idx):
        loss, logits_ebird, targets_ebird = self.model_step(batch, batch_idx)

        self.log(f"train/{self.loss.__class__.__name__}", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_metric_ebird(logits_ebird, targets_ebird.int())

        self.train_metric_combined.update(logits_ebird, targets_ebird.int())

        self.log(f"train/ebird_{self.train_metric_ebird.__class__.__name__}", self.train_metric_ebird, **asdict(self.logging_params))
        self.log_dict(self.train_metric_combined, **asdict(self.logging_params))
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, logits_ebird, targets_ebird = self.model_step(batch, batch_idx)

        self.log(f"val/{self.loss.__class__.__name__}", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_metric_ebird(logits_ebird, targets_ebird.int())

        self.val_metric_combined.update(logits_ebird, targets_ebird.int())

        self.log(f"val/ebird_{self.val_metric_ebird.__class__.__name__}", self.val_metric_ebird, **asdict(self.logging_params))
        self.log_dict(self.val_metric_combined, **asdict(self.logging_params))

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        loss, logits_ebird, targets_ebird = self.model_step(batch, batch_idx)

        self.log(f"test/{self.loss.__class__.__name__}", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_metric_ebird(logits_ebird, targets_ebird.int())

        self.test_metric_combined.update(logits_ebird, targets_ebird.int())

        self.log(f"test/ebird_{self.test_metric_ebird.__class__.__name__}", self.test_metric_ebird, **asdict(self.logging_params))
        self.log_dict(self.test_metric_combined, **asdict(self.logging_params))

        return {"loss": loss}

    def on_train_start(self):
        """
        Resets the best validation metrics at the beginning of training.
        """
        self.val_metric_best_ebird.reset()
        self.val_metric_best_combined_exact.reset()
        self.val_metric_best_combined_cmap.reset()

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

        # Combined task
        val_metric_combined = self.val_metric_combined.compute()
        self.val_metric_best_combined_exact.update(
            val_metric_combined["MultilabelExactMatch"]
        )
        self.log(
            f"val/combined_MultilabelExactMatch_best",
            self.val_metric_best_combined_exact.compute(),
            prog_bar=True,
        )

        self.val_metric_best_combined_cmap.update(val_metric_combined["cmAP"])
        self.log(
            f"val/combined_cmAP_best",
            self.val_metric_best_combined_cmap.compute(),
            prog_bar=True,
        )