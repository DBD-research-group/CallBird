from collections import OrderedDict, defaultdict
from functools import partial
from typing import Any

from omegaconf import AnyNode, DictConfig, ListConfig
from omegaconf.base import ContainerMetadata, Metadata
from torch import FloatStorage, LongStorage, device

# from torch import float32
from torch._C import _VariableFunctionsClass
from torch._C._nn import gelu # type: ignore
from torch._utils import _rebuild_parameter, _rebuild_tensor_v2
from torch.nn import ReLU
from torch.nn.modules.container import ModuleList, Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Identity, Linear
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.nn.modules.normalization import LayerNorm
from torch.optim.adamw import AdamW
from torch.serialization import add_safe_globals
from torchmetrics.aggregation import MaxMetric
from torchmetrics.classification.auroc import MultilabelAUROC
from torchmetrics.classification.average_precision import MultilabelAveragePrecision
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import jit_distributed_available
from torchmetrics.utilities.data import dim_zero_cat, dim_zero_max, dim_zero_sum
from transformers.activations import GELUActivation
from transformers.models.convnext.configuration_convnext import ConvNextConfig
from transformers.models.convnext.modeling_convnext import (
    ConvNextEmbeddings,
    ConvNextEncoder,
    ConvNextForImageClassification,
    ConvNextLayer,
    ConvNextLayerNorm,
    ConvNextModel,
    ConvNextStage,
)

from birdset.configs.module_configs import (
    LoggingParamsConfig,
    LRSchedulerConfig,
    MultilabelMetricsConfig,
)
from birdset.modules.metrics.multilabel import TopKAccuracy, cmAP, cmAP5, mAP, pcmAP
from birdset.modules.models.convnext import ConvNextClassifier
from projects.callbird.src.models.ConvNextMultiLayers import ConvNextMultiLayers, SingleLayerHead, ThreeLayerHead, TenLayerHead
from projects.callbird.src.models.ConvNextNoLayers import ConvNextNoLayers
from projects.callbird.src.models.ConvNextSameNoLayers import ConvNextSameNoLayers, SingleLayerHead as Sing


def ensure_torch_safe_globals():
    # Add DictConfig to the list of trusted types for torch.load
    add_safe_globals(
        [
            getattr,
            Sing, ConvNextSameNoLayers,
            dict,
            defaultdict,
            AdamW,
            Any,
            AnyNode,
            BCEWithLogitsLoss,
            ContainerMetadata,
            Conv2d,
            ConvNextClassifier,
            ConvNextConfig,
            ConvNextEmbeddings,
            ConvNextEncoder,
            ConvNextForImageClassification,
            ConvNextLayer,
            ConvNextLayerNorm,
            ConvNextModel,
            ConvNextStage,
            ConvNextNoLayers,
            DictConfig,
            FloatStorage,
            GELUActivation,
            Identity,
            LRSchedulerConfig,
            LayerNorm,
            Linear,
            LoggingParamsConfig,
            LongStorage,
            MaxMetric,
            Metadata,
            MetricCollection,
            ModuleList,
            MultilabelAUROC,
            MultilabelAveragePrecision,
            MultilabelMetricsConfig,
            OrderedDict,
            Sequential,
            TopKAccuracy,
            _VariableFunctionsClass,
            _rebuild_parameter,
            _rebuild_tensor_v2,
            cmAP,
            cmAP5,
            defaultdict,
            device,
            dim_zero_cat,
            dim_zero_max,
            dim_zero_sum,
            gelu,
            ConvNextMultiLayers,
            SingleLayerHead, ThreeLayerHead, TenLayerHead,
            ReLU,
            ListConfig,
            list,
            int,
            jit_distributed_available,
            mAP,
            partial,
            pcmAP,
        ]
    )
