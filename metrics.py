from typing import Optional

import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall, AUROC, AveragePrecision
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score, MultilabelPrecision,\
    MultilabelRecall, MultilabelAUROC, MultilabelAveragePrecision
from torchmetrics.segmentation import DiceScore, MeanIoU
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric

from einops import rearrange

class Metrics():
    def __init__(
        self,
        device: Optional[torch.device] = torch.device('cpu')
    ):
        self.device = device
        self.metrics = {}
    
    def update(self, pred: torch.Tensor, label: torch.Tensor) -> None:
        for name, metric in self.metrics.items():
            pred_proc, label_proc = self._prepare_inputs(name, pred, label)
            metric.update(pred_proc, label_proc)
        
    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor):
        # Default: do nothing
        return pred, label
    
    def compute(self) -> dict:
        metrics_dict = {}
        for metric_name, metric in self.metrics.items():
            metrics_dict[metric_name] = metric.compute()
        return metrics_dict
    
    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
    
    def get(self, name: str) -> float:
        if name not in self.metrics:
            raise ValueError(f"Metric {name} not found. Available: {list(self.metrics.keys())}")
        return self.metrics[name].compute()

class RegressionMetrics(Metrics):
    def __init__(
        self,
        device: Optional[torch.device] = torch.device('cpu'),
        num_outputs: int = None
    ):
        super().__init__(device)

        if num_outputs is None:
            self.num_outputs = 1
        else:
            self.num_outputs = num_outputs

        self.metrics = {
            'mae': MeanAbsoluteError().to(device),
            'mse': MeanSquaredError().to(device),
            'r2': R2Score(self.num_outputs, multioutput='uniform_average').to(device),
            'r2_per_output': R2Score(self.num_outputs, multioutput='raw_values').to(device),
            'pcc': PearsonCorrCoef(self.num_outputs).to(device)
        }

class BinaryClassificationMetrics(Metrics):
    def __init__(
        self,
        device = torch.device('cpu')
    ):
        super().__init__(device)

        self.metrics = {
            'accuracy': Accuracy(task='binary').to(device),
            'balanced_accuracy': Recall(task='binary', average='macro').to(device),
            'f1': F1Score(task='binary').to(device),
            'precision': Precision(task='binary').to(device),
            'recall': Recall(task='binary').to(device),
            'auroc': AUROC(task='binary').to(device),
            'auprc': AveragePrecision(task='binary').to(device),
        }
    
    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor): # pred: logits
        probs = torch.sigmoid(pred)
        label = label.long()

        if name in ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']:
            pred = (probs > 0.5).long()
        elif name in ['auroc', 'auprc']:
            pred = probs
        
        return pred, label

class MulticlassClassificationMetrics(Metrics):
    def __init__(
        self,
        device = torch.device('cpu'),
        num_classes: int = None
    ):
        super().__init__(device)

        if num_classes is None:
            self.num_classes = 1
        else:
            self.num_classes = num_classes

        self.metrics = {
            'accuracy': Accuracy(task='multiclass', average='macro', num_classes=self.num_classes).to(device), # balanced accuracy
            'f1': F1Score(task='multiclass', average='macro', num_classes=self.num_classes).to(device),
            'precision': Precision(task='multiclass', average='macro', num_classes=self.num_classes).to(device),
            'recall': Recall(task='multiclass', average='macro', num_classes=self.num_classes).to(device),
            'auroc': AUROC(task='multiclass', average='macro', num_classes=self.num_classes).to(device),
            'auprc': AveragePrecision(task='multiclass', average='macro', num_classes=self.num_classes).to(device),
        }
    
    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor): # pred: logits
        # logits -> probabilities
        probs = torch.softmax(pred, dim=-1)
        pred = probs
        label = label.long()
        label = label.argmax(dim=-1)

        return pred, label

class MultilabelClassificationMetrics(Metrics):
    def __init__(
        self,
        device = torch.device('cpu'),
        num_labels: int = None
    ):
        super().__init__(device)
        
        if num_labels is None:
            self.num_labels = 1
        else:
            self.num_labels = num_labels

        self.metrics = {
            'accuracy': MultilabelAccuracy(num_labels=self.num_labels, average='macro').to(device),
            'f1': MultilabelF1Score(num_labels=self.num_labels, average='macro').to(device),
            'precision': MultilabelPrecision(num_labels=self.num_labels, average='macro').to(device),
            'recall': MultilabelRecall(num_labels=self.num_labels, average='macro').to(device),
            'auroc': MultilabelAUROC(num_labels=self.num_labels, average='macro').to(device),
            'auprc': MultilabelAveragePrecision(num_labels=self.num_labels, average='macro').to(device),
        }
    
    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor):
        # logits -> probs via sigmoid
        probs = torch.sigmoid(pred)
        label = label.long()

        if name in ['accuracy', 'f1', 'precision', 'recall']:
            pred = (probs > 0.5).long()  # thresholded preds
        elif name in ['auroc', 'auprc']:
            pred = probs

        return pred, label

class SegmentationMetrics(Metrics):
    def __init__(
        self,
        device = torch.device('cpu'),
        num_classes: int = None
    ):
        super().__init__(device)

        if num_classes is None:
            self.num_classes = 1
        else:
            self.num_classes = num_classes
        
        self.metrics = {
            'dice': DiceScore(self.num_classes, include_background=False, average='macro', aggregation_level='samplewise').to(device),
            'iou': MeanIoU(self.num_classes, include_background=False, per_class=False).to(device),
            'hd95': HausdorffDistance(include_background=False, distance_metric='euclidean', reduction='mean').to(device),
            'assd': SurfaceDistance(include_background=False, distance_metric='euclidean', reduction='mean').to(device), 
            'dice_per_output': DiceScore(self.num_classes, include_background=False, average='none', aggregation_level='samplewise').to(device),
            'iou_per_output': MeanIoU(self.num_classes, include_background=False, per_class=True).to(device)
        }
    
    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor):
        """
        pred: (B, num_classes, T, H, W)
        label: (B, T, H, W) and contains class numbers
        """
        # convert logits into classes (one-hot)
        probs = torch.softmax(pred, dim=1) # (B, num_classes, T, H, W)
        pred = torch.argmax(probs, dim=1) # (B, T, H, W)
        pred_one_hot = F.one_hot(pred, num_classes=self.num_classes)
        pred_one_hot = rearrange(pred_one_hot, 'B T H W num_classes -> B num_classes T H W') # (B, num_classes, T, H, W)

        label_one_hot = F.one_hot(label, num_classes=self.num_classes)
        label_one_hot = rearrange(label_one_hot, 'B T H W num_classes -> B num_classes T H W') # (B, num_classes, T, H, W)

        return pred_one_hot, label_one_hot

class HausdorffDistance(Metric):
    """
    Wraps MONAI's HausdorffDistanceMetric as a torchmetrics metric
    because torchmetrics' HausdorffDistance is slow.
    """

    full_state_update: bool = False  # updates per batch

    def __init__(self, include_background: bool = False, distance_metric: str = 'euclidean', reduction: str = "mean", **kwargs):
        super().__init__(**kwargs)

        self.metric = HausdorffDistanceMetric(
            include_background=include_background,
            distance_metric=distance_metric,
            reduction="none"
        )

        # state for accumulation
        self.add_state("scores", default=[], dist_reduce_fx="cat")

        self.reduction = reduction

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        preds: (B, C, ...) predicted segmentation (one-hot or probabilities > thresholded)
        target: (B, C, ...) ground truth segmentation (one-hot encoded)
        """

        # MONAI expects preds and target as one-hot encoded
        # If preds are not one-hot, binarize them
        if preds.ndim == target.ndim - 1:  # e.g. missing channel
            preds = torch.nn.functional.one_hot(preds.long(), num_classes=target.shape[1])
            preds = preds.permute(0, -1, *range(1, preds.ndim-1))

        scores = self.metric(y_pred=preds, y=target)  # returns per-batch values
        self.scores.append(scores.detach().cpu())

    def compute(self):
        scores = torch.cat(self.scores, dim=0)

        if self.reduction == "mean":
            return scores.mean()
        elif self.reduction == "sum":
            return scores.sum()
        elif self.reduction == "none":
            return scores
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

class SurfaceDistance(Metric):
    """
    Wraps MONAI's SurfaceDistanceMetric as a torchmetrics metric.
    """

    full_state_update: bool = False  # updates per batch

    def __init__(self, include_background: bool = False, distance_metric: str = 'euclidean', reduction: str = "mean", symmetric: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.metric = SurfaceDistanceMetric(
            include_background=include_background,
            distance_metric=distance_metric,
            reduction="none",
            symmetric=symmetric
        )

        # state for accumulation
        self.add_state("scores", default=[], dist_reduce_fx="cat")

        self.reduction = reduction

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        preds: (B, C, ...) predicted segmentation (one-hot or probabilities > thresholded)
        target: (B, C, ...) ground truth segmentation (one-hot encoded)
        """

        # MONAI expects preds and target as one-hot encoded
        # If preds are not one-hot, binarize them
        if preds.ndim == target.ndim - 1:  # e.g. missing channel
            preds = torch.nn.functional.one_hot(preds.long(), num_classes=target.shape[1])
            preds = preds.permute(0, -1, *range(1, preds.ndim-1))

        scores = self.metric(y_pred=preds, y=target)  # returns per-batch values
        self.scores.append(scores.detach().cpu())

    def compute(self):
        scores = torch.cat(self.scores, dim=0)

        if self.reduction == "mean":
            return scores.mean()
        elif self.reduction == "sum":
            return scores.sum()
        elif self.reduction == "none":
            return scores
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


