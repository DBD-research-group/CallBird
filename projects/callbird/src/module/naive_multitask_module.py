from dataclasses import asdict
from typing import Type, Optional
import torch
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
        num_combined_classes: int,
        network: NetworkConfig = NetworkConfig(),
        loss: _Loss = BCEWithLogitsLoss(),
        optimizer: partial[Type[Optimizer]] = partial(AdamW, lr=1e-5),
        lr_scheduler: Optional[LRSchedulerConfig] = LRSchedulerConfig(),
        metrics_ebird: MultilabelMetricsConfig = MultilabelMetricsConfig(),
        metrics_combined: MetricCollection | None = None,
        logging_params: LoggingParamsConfig = LoggingParamsConfig(),
        debug: bool = False,
        **kwargs,
    ):
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
            metrics=None, # Explicitly pass None for metrics
            **kwargs,
        )
        self.logging_params = logging_params
        self.debug = debug

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

        # --- Debug instrumentation -------------------------------------------------
        if True:
            log_this_step = (self.global_step < 5) or (self.global_step % 500 == 0)
            if log_this_step:
                try:
                    with torch.no_grad():
                        # Basic input checks
                        if not torch.isfinite(input_values).all():
                            print(
                                f"[DEBUG][step={self.global_step}] Non-finite input_values detected: "
                                f"nan={(~torch.isfinite(input_values)).logical_and(torch.isnan(input_values)).sum().item()} "
                                f"inf={torch.isinf(input_values).sum().item()}"
                            )
                        num_ebird_classes = logits_ebird.shape[-1]
                        max_label_val = labels_ebird.max().item()
                        min_label_val = labels_ebird.min().item()
                        positives_per_sample = labels_ebird.sum(dim=-1)
                        max_pos = positives_per_sample.max().item()
                        mean_pos = positives_per_sample.float().mean().item()
                        gt_over_one = (labels_ebird > 1).sum().item()
                        # Logits stats
                        logits_min = logits_ebird.min().item()
                        logits_max = logits_ebird.max().item()
                        logits_mean = logits_ebird.mean().item()
                        logits_std = logits_ebird.std().item()
                        logits_abs_max = logits_ebird.abs().max().item()
                        extreme_50 = (logits_ebird.abs() > 50).float().mean().item()
                        extreme_80 = (logits_ebird.abs() > 80).float().mean().item()
                        non_finite_logits = (~torch.isfinite(logits_ebird)).sum().item()
                        msg = (
                            f"[DEBUG][step={self.global_step}] ebird logits shape={tuple(logits_ebird.shape)} "
                            f"num_classes={num_ebird_classes} labels_range=[{min_label_val},{max_label_val}] "
                            f"mean_pos={mean_pos:.2f} max_pos={max_pos} count_vals_gt_1={gt_over_one} "
                            f"logits[min={logits_min:.2f},max={logits_max:.2f},mean={logits_mean:.2f},std={logits_std:.2f},|max|={logits_abs_max:.2f}] "
                            f"extreme>|50|={extreme_50*100:.2f}% extreme>|80|={extreme_80*100:.2f}% non_finite={non_finite_logits}"
                        )
                        if 'labels_calltype' in batch:
                            labels_call = batch['labels_calltype']
                            num_call_classes = labels_call.shape[-1]
                            call_max = labels_call.max().item()
                            call_min = labels_call.min().item()
                            call_over_one = (labels_call > 1).sum().item()
                            call_mean_pos = labels_call.sum(dim=-1).float().mean().item()
                            msg += (
                                f" | calltype shape={tuple(labels_call.shape)} num_classes={num_call_classes} "
                                f"range=[{call_min},{call_max}] mean_pos={call_mean_pos:.2f} gt_1={call_over_one}"
                            )
                        # Use Lightning logging if available, else print
                        if hasattr(self.logger, 'log'):
                            try:
                                self.logger.log(msg)
                            except Exception:
                                print(msg)
                        else:
                            print(msg)
                        # Hard assert to catch out-of-range targets early
                        assert max_label_val <= 1 and min_label_val >= 0, (
                            "Out-of-range ebird labels detected (not in [0,1]); check merge mapping."
                        )
                except Exception as debug_exc:
                    print(f"[DEBUG] Failed to collect debug stats: {debug_exc}")
        # --------------------------------------------------------------------------

        loss_ebird = self.loss(logits_ebird, labels_ebird)

        if True and torch.isnan(loss_ebird):
            print("[DEBUG] NaN loss encountered. Collecting diagnostics...")
            with torch.no_grad():
                finite_mask = torch.isfinite(logits_ebird)
                if not finite_mask.all():
                    print(f"[DEBUG] Non-finite logits count: {(~finite_mask).sum().item()}")
                print(f"[DEBUG] logits abs max: {logits_ebird.abs().max().item():.4f}")
                print(f"[DEBUG] logits sample (first row, first 10): {logits_ebird[0, :10].tolist()}")
                print(f"[DEBUG] labels unique values: {torch.unique(labels_ebird)}")
                nan_params = []
                inf_params = []
                for name, p in self.named_parameters():
                    if p.requires_grad:
                        if torch.isnan(p).any():
                            nan_params.append(name)
                        if torch.isinf(p).any():
                            inf_params.append(name)
                if nan_params:
                    print(f"[DEBUG] Parameters containing NaN: {nan_params[:10]}{' (truncated)' if len(nan_params)>10 else ''}")
                if inf_params:
                    print(f"[DEBUG] Parameters containing Inf: {inf_params[:10]}{' (truncated)' if len(inf_params)>10 else ''}")
                for name, p in list(self.named_parameters())[:5]:
                    if p.requires_grad and torch.isfinite(p).all():
                        print(
                            f"[DEBUG] Param {name}: mean={p.data.mean():.4e} std={p.data.std():.4e} absmax={p.data.abs().max():.4e}"
                        )
            raise RuntimeError("NaN loss detected; see debug diagnostics above.")
        
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