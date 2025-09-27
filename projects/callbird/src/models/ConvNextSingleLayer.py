from birdset import utils
from birdset.modules.models.convnext import ConvNextClassifier
from typing import Optional, Dict
import torch
import torch.nn as nn

log = utils.get_pylogger(__name__)

class SingleLayerHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class ConvNextSingleLayer(nn.Module):

    def __init__(self, num_combined_classes: int, freeze_backbone: bool, **kwargs):
        """
        Initializes the ConvNextSingleLayer model for two tasks.

        Args:
            num_classes (int): The number of output classes for each task.
            hidden_dim (int): The dimension of the hidden layers in the task heads.
            **kwargs: Additional keyword arguments for convnext.
        """
        super().__init__()

        local_checkpoint = kwargs.pop("local_checkpoint", None)

        self.convnext = ConvNextClassifier(
            num_classes=9736,
            num_channels=1,
            checkpoint= "DBD-research-group/ConvNeXT-Base-BirdSet-XCL" if local_checkpoint is None else None,
            local_checkpoint=None,
            cache_dir=None,
            pretrain_info=None
        )

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

        # Get the pooled output features from the base ConvNext model
        pooled_output = self.convnext(input_values)

        # Get logits for each task by passing the features through the respective heads
        logits_ebird = self.ebird_head(pooled_output)

        return {
            "ebird_code": logits_ebird,
        }
    
    def freeze_model_backbone(self):
        """
        Freezes the backbone of the model.
        """
        for param in self.convnext.model.parameters():
            param.requires_grad = False
        log.info(">> Backbone of the model is frozen.")