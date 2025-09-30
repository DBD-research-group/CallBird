from dataclasses import asdict
from typing import Type, Optional
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer
from functools import partial
import torch
from torchmetrics import Accuracy, MetricCollection

from birdset.modules.base_module import BaseModule
from birdset.configs import (
    NetworkConfig,
    LRSchedulerConfig,
    MultilabelMetricsConfig,
    LoggingParamsConfig,
)
from birdset.modules.metrics.multilabel import cmAP
from torchmetrics.classification import MultilabelAccuracy, MultilabelExactMatch


class MultiTaskModule(BaseModule):
    def __init__(
        self,
        num_combined_classes: int,
        network: NetworkConfig = NetworkConfig(),
        loss: _Loss = BCEWithLogitsLoss(),
        optimizer: partial[Type[Optimizer]] = partial(AdamW, lr=1e-5),
        lr_scheduler: Optional[LRSchedulerConfig] = LRSchedulerConfig(),
        metrics_ebird: MultilabelMetricsConfig = MultilabelMetricsConfig(),
        metrics_calltype: MultilabelMetricsConfig = MultilabelMetricsConfig(),
        metrics_combined: MetricCollection | None = None,
        logging_params: LoggingParamsConfig = LoggingParamsConfig(),
        **kwargs,
    ):
        self.num_combined_classes = num_combined_classes
        # Placeholder for inferred mapping list[(ebird_idx, calltype_idx)] aligned with labels_combined ordering.
        self._combined_mapping: Optional[list[tuple[int, int]]] = None
        
        if metrics_combined is None:
            metrics_combined = MetricCollection(
            [
                cmAP(num_labels=num_combined_classes),
                MultilabelAccuracy(num_labels=num_combined_classes),
                MultilabelExactMatch(num_labels=num_combined_classes),
            ]
        )

        # Remove the unexpected argument before calling the parent constructor
        kwargs.pop("prediction_table", None)
        # We are handling metrics ourselves, so remove it before calling super
        kwargs.pop("metrics", None)

        super().__init__(
            network=network,
            loss=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=None,  # Explicitly pass None for metrics
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
        input_values, labels_ebird, labels_calltype = batch["input_values"], batch["labels_ebird"], batch["labels_calltype"]
        
        outputs = self.forward(input_values=input_values)
        logits_ebird = outputs["ebird_code"]
        logits_calltype = outputs["call_type"]

        loss_ebird = self.loss(logits_ebird, labels_ebird)
        loss_calltype = self.loss(logits_calltype, labels_calltype)
        
        total_loss = loss_ebird + loss_calltype

        return total_loss, logits_ebird, labels_ebird, logits_calltype, labels_calltype

    # -------------------------------- Mapping & Joint Logic ---------------------------------
    def _ensure_combined_mapping(self):
        """Infer (ebird_idx, calltype_idx) per combined class.

        Simplified assumption (per dataset construction): combined label strings were created as
            f"{ebird_code}_{call_type}"
        where ebird_code itself never contains an underscore, but call_type MAY contain underscores.
        Therefore we must split at the FIRST underscore, not the last one.

        Requirements:
          - data_module.combined_labels: list[str] with ordering matching one-hot encoding of labels_combined
          - data_module.ebird_labels / calltype_labels: species and call type vocab lists
        """
        if self._combined_mapping is not None:
            return
        # Access the datamodule lazily through trainer (ensures setup has run)
        if self.trainer is None or self.trainer.datamodule is None:
            raise RuntimeError("Trainer or datamodule not attached yet; mapping requested too early.")
        dm = self.trainer.datamodule
        if not all(hasattr(dm, attr) for attr in ["combined_labels", "ebird_labels", "calltype_labels"]):
            raise RuntimeError(
                "Datamodule must define combined_labels, ebird_labels, calltype_labels after setup before mapping can be built."
            )
        ebird_index = {lbl: i for i, lbl in enumerate(dm.ebird_labels)}
        call_index = {lbl: i for i, lbl in enumerate(dm.calltype_labels)}
        mapping: list[tuple[int, int]] = []
        for combined in dm.combined_labels:
            if "_" not in combined:
                raise ValueError(
                    f"Combined label '{combined}' missing '_' delimiter expected from construction."
                )
            # Split at the FIRST underscore: species code first token, remainder is full call type (may contain underscores)
            sp, ct = combined.split("_", 1)
            if sp not in ebird_index:
                raise ValueError(
                    f"Species part '{sp}' from combined label '{combined}' not found in ebird vocabulary."
                )
            if ct not in call_index:
                raise ValueError(
                    f"Call type part '{ct}' from combined label '{combined}' not found in call type vocabulary."
                )
            mapping.append((ebird_index[sp], call_index[ct]))

        if len(mapping) != self.num_combined_classes:
            raise ValueError(
                f"Inferred mapping length {len(mapping)} != num_combined_classes {self.num_combined_classes}"
            )
        self._combined_mapping = mapping

    def _joint_logits_and_targets(
        self,
        logits_ebird: torch.Tensor,
        logits_calltype: torch.Tensor,
        targets_ebird: torch.Tensor,
        targets_calltype: torch.Tensor,
        batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return joint (combined) logits and targets aligned with labels_combined ordering.

        If batch contains 'labels_combined' we take those as the target multi-hot vector. Otherwise we derive
        targets via logical AND of the individual heads at mapped indices.
        Joint probability p_combo = sigmoid(l_e[i]) * sigmoid(l_c[j]); transformed back to logits for metric
        compatibility with naive combined head.
        """
        self._ensure_combined_mapping()
        mapping = self._combined_mapping  # type: ignore
        probs_e = torch.sigmoid(logits_ebird)
        probs_c = torch.sigmoid(logits_calltype)
        device = probs_e.device
        ebird_idx = torch.as_tensor([m[0] for m in mapping], device=device)
        call_idx = torch.as_tensor([m[1] for m in mapping], device=device)
        joint_probs = probs_e[:, ebird_idx] * probs_c[:, call_idx]
        joint_logits = torch.logit(joint_probs.clamp(1e-6, 1 - 1e-6))

        if "labels_combined" in batch:
            joint_targets = batch["labels_combined"].int()
        else:
            tgt_e = targets_ebird.int()[:, ebird_idx]
            tgt_c = targets_calltype.int()[:, call_idx]
            joint_targets = (tgt_e & tgt_c).int()
        return joint_logits, joint_targets

    def training_step(self, batch, batch_idx):
        loss, logits_ebird, targets_ebird, logits_calltype, targets_calltype = self.model_step(batch, batch_idx)

        self.log(f"train/{self.loss.__class__.__name__}", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_metric_ebird(logits_ebird, targets_ebird.int())
        self.train_metric_calltype(logits_calltype, targets_calltype.int())
        joint_logits, joint_targets = self._joint_logits_and_targets(
            logits_ebird, logits_calltype, targets_ebird, targets_calltype, batch
        )
        self.train_metric_combined.update(joint_logits, joint_targets)

        self.log(f"train/ebird_{self.train_metric_ebird.__class__.__name__}", self.train_metric_ebird, **asdict(self.logging_params))
        self.log(f"train/calltype_{self.train_metric_calltype.__class__.__name__}", self.train_metric_calltype, **asdict(self.logging_params))
        self.log_dict(self.train_metric_combined, **asdict(self.logging_params))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, logits_ebird, targets_ebird, logits_calltype, targets_calltype = self.model_step(batch, batch_idx)

        self.log(f"val/{self.loss.__class__.__name__}", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_metric_ebird(logits_ebird, targets_ebird.int())
        self.val_metric_calltype(logits_calltype, targets_calltype.int())
        joint_logits, joint_targets = self._joint_logits_and_targets(
            logits_ebird, logits_calltype, targets_ebird, targets_calltype, batch
        )
        self.val_metric_combined.update(joint_logits, joint_targets)

        self.log(f"val/ebird_{self.val_metric_ebird.__class__.__name__}", self.val_metric_ebird, **asdict(self.logging_params))
        self.log(f"val/calltype_{self.val_metric_calltype.__class__.__name__}", self.val_metric_calltype, **asdict(self.logging_params))
        self.log_dict(self.val_metric_combined, **asdict(self.logging_params))

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        loss, logits_ebird, targets_ebird, logits_calltype, targets_calltype = self.model_step(batch, batch_idx)

        self.log(f"test/{self.loss.__class__.__name__}", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_metric_ebird(logits_ebird, targets_ebird.int())
        self.test_metric_calltype(logits_calltype, targets_calltype.int())
        joint_logits, joint_targets = self._joint_logits_and_targets(
            logits_ebird, logits_calltype, targets_ebird, targets_calltype, batch
        )
        self.test_metric_combined.update(joint_logits, joint_targets)

        self.log(f"test/ebird_{self.test_metric_ebird.__class__.__name__}", self.test_metric_ebird, **asdict(self.logging_params))
        self.log(f"test/calltype_{self.test_metric_calltype.__class__.__name__}", self.test_metric_calltype, **asdict(self.logging_params))
        self.log_dict(self.test_metric_combined, **asdict(self.logging_params))

        return {"loss": loss}

    def on_train_start(self):
        """
        Resets the best validation metrics at the beginning of training.
        """
        self.val_metric_best_ebird.reset()
        self.val_metric_best_calltype.reset()
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