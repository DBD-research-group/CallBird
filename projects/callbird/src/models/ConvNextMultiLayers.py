from birdset import utils
from birdset.modules.models.convnext import ConvNextClassifier
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

class ConvNextMultiLayers(nn.Module):

    def __init__(self, freeze_backbone: bool = False, variant: str = "three", **kwargs):
        """
        Initializes the ConvNextMultiLayers model for two tasks.

        Args:
            num_classes (int): The number of output classes for each task.
            hidden_dim (int): The dimension of the hidden layers in the task heads.
            **kwargs: Additional keyword arguments for convnext.
        """
        super().__init__()

        local_checkpoint = kwargs.pop("local_checkpoint", None)

        self.convnext = ConvNextClassifier(
            num_classes=1000,
            num_channels=1,
            checkpoint="DBD-research-group/ConvNeXT-Base-BirdSet-XCL",
            local_checkpoint=None,
            cache_dir=None,
            pretrain_info=None
        )

        if freeze_backbone:
            self.freeze_model_backbone()

        # Get the number of input features from the original classifier
        in_features = self.convnext.model.classifier.in_features

        # Replace the original classifier with an identity layer
        self.convnext.model.classifier = nn.Identity()

        if variant == "single":
            self.ebird_head = SingleLayerHead(in_features, 54)
            self.calltype_head = SingleLayerHead(in_features, 7)
        elif variant == "three":
            self.ebird_head = ThreeLayerHead(in_features, 54)
            self.calltype_head = ThreeLayerHead(in_features, 7)
        elif variant == "ten":
            self.ebird_head = TenLayerHead(in_features, 54)
            self.calltype_head = TenLayerHead(in_features, 7)
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
        state_dict = torch.load(checkpoint_path)["state_dict"]
        
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
        logits_calltype = self.calltype_head(pooled_output)


        return {
            "ebird_code": logits_ebird,
            "call_type": logits_calltype,
        }
    
    def freeze_model_backbone(self):
        """
        Freezes the backbone of the model.
        """
        for param in self.convnext.model.parameters():
            param.requires_grad = False
        log.info(">> Backbone of the model is frozen.")