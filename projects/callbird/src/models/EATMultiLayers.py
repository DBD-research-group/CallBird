from birdset import utils
from typing import Optional, Dict
import torch
import torch.nn as nn

class SingleLayerHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class ThreeLayerHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class TenLayerHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

log = utils.get_pylogger(__name__)

from birdset.modules.models.eat_soundnet import SoundNet
from typing import List

class EATMultiLayers(nn.Module):

    def __init__(
        self,
        freeze_backbone: bool,
        variant: str,
        num_classes_calltype: int,
        num_classes_ebird: int,
        num_combined_classes: int,
        # EAT parameters directly from SoundNet class constructor
        nf: int = 32,
        seq_len: int = 90112,
        embed_dim: int = 128,
        n_layers: int = 4,
        nhead: int = 8,
        factors: List[int] = [4, 4, 4, 4],
        dim_feedforward: int = 512,
        local_checkpoint: str | None = None,
        device: str = "cuda:0",
        num_classes: int | None = None,  # it is set by default by convnext, so we need to ignore it here (to avoid errors)
        num_channels: int | None = None,  # it is set by default by convnext, so we need to ignore it here (to avoid errors)
        checkpoint: str | None = None,  # it is set by default by convnext, so we need to ignore it here (to avoid errors)
        cache_dir: str | None = None,  # it is set by default by convnext, so we need to ignore it here (to avoid errors)
        pretrain_info: str | None = None,  # it is set by default by convnext, so we need to ignore it here (to avoid errors)
        **kwargs
    ):
        """
        Initializes the EATMultiLayers model for two tasks.

        Args:
            num_classes (int): The number of output classes for each task.
            hidden_dim (int): The dimension of the hidden layers in the task heads.
            **kwargs: Additional keyword arguments for convnext.
        """
        super().__init__()

        # local_checkpoint = kwargs.pop("local_checkpoint", None)

        self.eat = SoundNet(
            nf = nf,
            seq_len = seq_len,
            embed_dim = embed_dim,
            n_layers = n_layers,
            nhead = nhead,
            factors = factors,
            dim_feedforward = dim_feedforward,
            device = device,

            num_classes=9736,
            # "DBD-research-group/ConvNeXT-Base-BirdSet-XCL" if local_checkpoint is None else None,
            local_checkpoint="/workspace/projects/callbird/src/models/eat-checkpoint-20.ckpt" if local_checkpoint is None else None,
            pretrain_info=None
        )

        if freeze_backbone:
            self.freeze_model_backbone()

        # Get the number of input features from the original classifier
        in_features = self.eat.tf.fc.in_features

        # Replace the original classifier with an identity layer
        self.eat.tf.fc = nn.Identity()

        if variant == "single":
            self.ebird_head = SingleLayerHead(in_features, num_classes_ebird)
            self.calltype_head = SingleLayerHead(in_features, num_classes_calltype)
        elif variant == "three":
            self.ebird_head = ThreeLayerHead(in_features, num_classes_ebird)
            self.calltype_head = ThreeLayerHead(in_features, num_classes_calltype)
        elif variant == "ten":
            self.ebird_head = TenLayerHead(in_features, num_classes_ebird)
            self.calltype_head = TenLayerHead(in_features, num_classes_calltype)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        if local_checkpoint:
            self.load_from_checkpoint(local_checkpoint)

    def load_from_checkpoint(self, checkpoint_path: str):
        """
        Loads weights from a local checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        ckpt = torch.load(checkpoint_path, weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        # Adjust keys if they are prefixed (e.g., by a LightningModule)
        adjusted_state_dict = {}
        for key, value in state_dict.items():
            # Drop uncertainty weighting params from MultiTaskModule
            if "log_var_ebird" in key or "log_var_calltype" in key:
                continue
            new_key = key.replace("model.", "", 1)  # remove 'model.' prefix once if present
            adjusted_state_dict[new_key] = value

        # Load non-strictly to ignore any remaining non-matching keys
        self.load_state_dict(adjusted_state_dict, strict=False)
        log.info(f"Loaded model weights from {checkpoint_path} (filtered uncertainty params)")

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:

        # Get the pooled output features from the base EAT model
        pooled_output = self.eat(input_values)

        # Get logits for each task by passing the features through the respective heads
        logits_ebird = self.ebird_head(pooled_output)
        logits_calltype = self.calltype_head(pooled_output)


        return {
            "ebird_code": logits_ebird,
            "call_type": logits_calltype,
        }
    
    def freeze_model_backbone(self):
        """
        Freezes the backbone of the model.
        """
        for param in self.eat.parameters():
            param.requires_grad = False
        log.info(">> Backbone of the model is frozen.")