from dataclasses import asdict
from typing import Type, Optional
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer
from functools import partial
import torch
import torch.nn as nn
import math
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
        # If True, use homoscedastic uncertainty to dynamically weight task losses
        dynamic_loss: bool = True,
        **kwargs,
    ):
        self.num_combined_classes = num_combined_classes
        self.use_uncertainty_weighting = dynamic_loss
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

        # Learnable log-variances for homoscedastic uncertainty weighting (per-task)
        # s = log(sigma^2); effective weight = exp(-s)
        if self.use_uncertainty_weighting:
            self.log_var_ebird = nn.Parameter(torch.zeros(1))
            self.log_var_calltype = nn.Parameter(torch.zeros(1))

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

    # ------------------------------- Optimizer & Checkpointing -------------------------------
    def configure_optimizers(self):
        """Ensure uncertainty parameters are optimized along with the model, mirroring BaseModule scheduling."""
        # Model params group
        param_groups = [{"params": self.model.parameters()}]
        # Add uncertainty params without weight decay
        if getattr(self, "use_uncertainty_weighting", False):
            param_groups.append({
                "params": [self.log_var_ebird, self.log_var_calltype],
                "weight_decay": 0.0,
            })

        self.optimizer = self.optimizer(param_groups)
        if self.lr_scheduler is not None:
            num_training_steps = math.ceil(
                (self.num_epochs * self.len_trainset) / self.batch_size * self.num_gpus
            )
            num_warmup_steps = math.ceil(
                num_training_steps * self.lr_scheduler.warmup_ratio
            )
            self.scheduler = self.lr_scheduler.scheduler(
                optimizer=self.optimizer,
                num_training_steps=num_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
            scheduler_dict = {
                "scheduler": self.scheduler,
                "interval": self.lr_scheduler.interval,
                "warmup_ratio": self.lr_scheduler.warmup_ratio,
            }
            return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}

        return {"optimizer": self.optimizer}

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Allow loading checkpoints that predate the uncertainty params.

        Falls back to non-strict to skip missing/extra keys for log_var_* while loading everything else.
        """
        result = super().load_state_dict(state_dict, strict=False)
        # Optionally log info for visibility
        missing = getattr(result, 'missing_keys', []) if hasattr(result, 'missing_keys') else []
        unexpected = getattr(result, 'unexpected_keys', []) if hasattr(result, 'unexpected_keys') else []
        if missing or unexpected:
            try:
                self.print(f"load_state_dict: missing={missing}, unexpected={unexpected}")
            except Exception:
                pass
        return result

    def model_step(self, batch, batch_idx):
        input_values, labels_ebird, labels_calltype = batch["input_values"], batch["labels_ebird"], batch["labels_calltype"]
        
        outputs = self.forward(input_values=input_values)
        logits_ebird = outputs["ebird_code"]
        logits_calltype = outputs["call_type"]

        loss_ebird = self.loss(logits_ebird, labels_ebird)
        loss_calltype = self.loss(logits_calltype, labels_calltype)

        # Dynamic weighting via homoscedastic uncertainty (Kendall et al. 2018)
        if getattr(self, "use_uncertainty_weighting", False):
            # Clamp to keep weights and the additive log-term in a sane range
            s_e = self.log_var_ebird.clamp(min=-10.0, max=10.0)
            s_c = self.log_var_calltype.clamp(min=-10.0, max=10.0)
            # Paper-aligned per-task: exp(-s_i) * L_i + 0.5 * s_i  where s_i = log(sigma_i^2)
            total_loss = torch.exp(-s_e) * loss_ebird + 0.5 * s_e
            total_loss = total_loss + torch.exp(-s_c) * loss_calltype + 0.5 * s_c
        else:
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
        # Optionally log effective task weights for monitoring
        if getattr(self, "use_uncertainty_weighting", False):
            # Effective weights: w = exp(-s), with clamped s for stability in logs
            s_e = self.log_var_ebird.detach().clamp(min=-10.0, max=10.0)
            s_c = self.log_var_calltype.detach().clamp(min=-10.0, max=10.0)
            w_e = torch.exp(-s_e)
            w_c = torch.exp(-s_c)
            self.log("train/weight_ebird", w_e.item(), on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/weight_calltype", w_c.item(), on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/s_ebird", s_e.item(), on_step=False, on_epoch=True, prog_bar=False)
            self.log("train/s_calltype", s_c.item(), on_step=False, on_epoch=True, prog_bar=False)
        # Log raw (unweighted) sum of task losses for reference (always non-negative)
        raw_sum = self.loss(logits_ebird, targets_ebird) + self.loss(logits_calltype, targets_calltype)
        self.log("train/raw_loss_sum", raw_sum, on_step=False, on_epoch=True, prog_bar=False)
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
        if getattr(self, "use_uncertainty_weighting", False):
            s_e = self.log_var_ebird.detach().clamp(min=-10.0, max=10.0)
            s_c = self.log_var_calltype.detach().clamp(min=-10.0, max=10.0)
            w_e = torch.exp(-s_e)
            w_c = torch.exp(-s_c)
            self.log("val/weight_ebird", w_e.item(), on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/weight_calltype", w_c.item(), on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/s_ebird", s_e.item(), on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/s_calltype", s_c.item(), on_step=False, on_epoch=True, prog_bar=False)
        raw_sum = self.loss(logits_ebird, targets_ebird) + self.loss(logits_calltype, targets_calltype)
        self.log("val/raw_loss_sum", raw_sum, on_step=False, on_epoch=True, prog_bar=False)
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
        if getattr(self, "use_uncertainty_weighting", False):
            s_e = self.log_var_ebird.detach().clamp(min=-10.0, max=10.0)
            s_c = self.log_var_calltype.detach().clamp(min=-10.0, max=10.0)
            w_e = torch.exp(-s_e)
            w_c = torch.exp(-s_c)
            self.log("test/weight_ebird", w_e.item(), on_step=False, on_epoch=True, prog_bar=False)
            self.log("test/weight_calltype", w_c.item(), on_step=False, on_epoch=True, prog_bar=False)
            self.log("test/s_ebird", s_e.item(), on_step=False, on_epoch=True, prog_bar=False)
            self.log("test/s_calltype", s_c.item(), on_step=False, on_epoch=True, prog_bar=False)
        raw_sum = self.loss(logits_ebird, targets_ebird) + self.loss(logits_calltype, targets_calltype)
        self.log("test/raw_loss_sum", raw_sum, on_step=False, on_epoch=True, prog_bar=False)
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