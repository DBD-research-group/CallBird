from birdset import utils
from birdset.modules.models.eat_soundnet import SoundNet
from typing import Optional, Dict
import torch
import torch.nn as nn
from typing import List

log = utils.get_pylogger(__name__)

class SingleLayerHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class EATSingleLayer(nn.Module):

    def __init__(
        self,
        num_combined_classes: int,
        freeze_backbone: bool,
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
    ):
        """
        Initializes the EATSingleLayer model for two tasks.

        Args:
            num_classes (int): The number of output classes for each task.
            hidden_dim (int): The dimension of the hidden layers in the task heads.
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

        # Replace the original classifier with an identity layer
        self.eat.classifier = nn.Identity()

        self.ebird_head = SingleLayerHead(9736, num_combined_classes)
        
        if local_checkpoint is not None:
            self.load_from_checkpoint(local_checkpoint)

        if freeze_backbone:
            self.freeze_model_backbone()

    def load_from_checkpoint(self, checkpoint_path: str):
        """
        Loads weights from a local checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """

        if checkpoint_path is None:
            raise Error("Checkpoint path must be provided to load model weights.")

        state_dict = torch.load(checkpoint_path, weights_only=False)["state_dict"]
        
        # Adjust keys if they are prefixed (e.g., by a LightningModule)
        adjusted_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("model.", "", 1) # remove 'model.' prefix
            adjusted_state_dict[new_key] = value
            
        self.load_state_dict(adjusted_state_dict)
        log.info(f"Loaded model weights from {checkpoint_path}")

    def forward(
        self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:

        # Get the pooled output features from the base EAT model
        pooled_output = self.eat(input_values)

        # Get logits for each task by passing the features through the respective heads
        logits_ebird = self.ebird_head(pooled_output)

        return {
            "ebird_code": logits_ebird,
        }
    
    def freeze_model_backbone(self):
        """
        Freezes the backbone of the model.
        """
        for param in self.eat.parameters():
            param.requires_grad = False
        log.info(">> Backbone of the model is frozen.")