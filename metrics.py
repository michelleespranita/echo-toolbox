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
        self._metric_has_data = {name: False for name, metric in self.metrics.items()}
    
    def update(self, pred: torch.Tensor, label: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        for name, metric in self.metrics.items():
            pred_proc, label_proc = self._prepare_inputs(name, pred, label, mask)
            if pred_proc is None:
                continue
            metric.update(pred_proc, label_proc)
            self._metric_has_data[name] = True
        
    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Default: do nothing
        return pred, label

    def _safe_compute_metric(self, metric: Metric):
        try:
            return metric.compute()
        except ValueError as e:
            error_msg = str(e).lower()
            # Some regression/correlation metrics require at least 2 samples.
            if 'at least two samples' in error_msg:
                return float('nan')
            raise
    
    def compute(self) -> dict:
        metrics_dict = {}
        for metric_name, metric in self.metrics.items():
            metrics_dict[metric_name] = self._safe_compute_metric(metric) if self._metric_has_data[metric_name] else float('nan')
        return metrics_dict
    
    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
        self._metric_has_data = {name: False for name in self._metric_has_data}
    
    def get(self, name: str) -> float:
        if name not in self.metrics:
            raise ValueError(f"Metric {name} not found. Available: {list(self.metrics.keys())}")
        return self._safe_compute_metric(self.metrics[name])

class RegressionMetrics(Metrics):
    def __init__(
        self,
        device: Optional[torch.device] = torch.device('cpu'),
        num_outputs: Optional[int] = None
    ):
        super().__init__(device)

        if num_outputs is None:
            self.num_outputs = 1
        else:
            self.num_outputs = num_outputs

        self.metrics = {
            'mae': MeanAbsoluteError().to(device),
            'mse': MeanSquaredError().to(device),
            'r2': R2Score(multioutput='uniform_average').to(device), # num_outputs is not automatically inferred from the shape of input tensors
            'pcc': PearsonCorrCoef(num_outputs=self.num_outputs).to(device)
        }
        self._metric_has_data = {name: False for name, metric in self.metrics.items()}

        if num_outputs > 1:
            self.metrics['r2_per_output'] = R2Score(multioutput='raw_values').to(device)
    
    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        mask: 1 for NaN (ignore), 0 for valid (calculate).
        """
        if mask is None:
            if not (label == -100).any():
                return pred, label
            mask = ~(label == -100) # (B, num_outputs). True where valid, False where invalid
        else:
            mask = ~(mask.bool()) # True where valid, False where invalid
        
        if mask.sum() < 2: # not enough valid samples to compute metric, e.g. r2 requires at least 2 samples
            return None, None # handled by Metrics class
        
        pred_valid = pred[mask]
        label_valid = label[mask]

        # R² and PCC are undefined when all labels are identical
        if name in ('r2', 'r2_per_output', 'pcc'):
            if label_valid.unique().numel() == 1:
                return None, None  # will produce nan in compute()

        return pred_valid, label_valid # (num_valid,)

class BinaryClassificationMetrics(Metrics):
    def __init__(
        self,
        device = torch.device('cpu')
    ):
        super().__init__(device)

        self.metrics = {
            'accuracy': Accuracy(task='binary').to(device),
            'balanced_accuracy': Accuracy(task='multiclass', average='macro', num_classes=2).to(device),
            'f1': F1Score(task='binary').to(device),
            'precision': Precision(task='binary').to(device),
            'recall': Recall(task='binary').to(device),
            'auroc': AUROC(task='binary').to(device),
            'auprc': AveragePrecision(task='binary').to(device),
        }
        self._metric_has_data = {name: False for name, metric in self.metrics.items()}
    
    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor, mask: Optional[torch.Tensor] = None): # pred: logits
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
            'accuracy': Accuracy(task='multiclass', average='micro', num_classes=self.num_classes).to(device),
            'balanced_accuracy': Accuracy(task='multiclass', average='macro', num_classes=self.num_classes).to(device),
            'f1': F1Score(task='multiclass', average='macro', num_classes=self.num_classes).to(device),
            'precision': Precision(task='multiclass', average='macro', num_classes=self.num_classes).to(device),
            'recall': Recall(task='multiclass', average='macro', num_classes=self.num_classes).to(device),
            'auroc': AUROC(task='multiclass', average='macro', num_classes=self.num_classes).to(device),
            'auprc': AveragePrecision(task='multiclass', average='macro', num_classes=self.num_classes).to(device),
        }
        self._metric_has_data = {name: False for name, metric in self.metrics.items()}
    
    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor, mask: Optional[torch.Tensor] = None): # pred: logits
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
            'auroc_per_output': MultilabelAUROC(num_labels=self.num_labels, average='none').to(device),
            'auprc': MultilabelAveragePrecision(num_labels=self.num_labels, average='macro').to(device),
        }
        self._metric_has_data = {name: False for name, metric in self.metrics.items()}
    
    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # logits -> probs via sigmoid
        probs = torch.sigmoid(pred)
        label = label.long()

        if name in ['accuracy', 'f1', 'precision', 'recall']:
            pred = (probs > 0.5).long()  # thresholded preds
        elif name in ['auroc', 'auprc']:
            pred = probs

        return pred, label

class MultiHeadBinaryClassificationMetrics(Metrics):
    def __init__(
        self,
        device = torch.device('cpu'),
        num_tasks: int = None
    ):
        super().__init__(device)
        
        if num_tasks is None:
            self.num_tasks = 1
        else:
            self.num_tasks = num_tasks
        
        self.task_names = [i for i in range(self.num_tasks)]

        self.metrics = {
            # 'accuracy': MultilabelAccuracy(num_labels=self.num_tasks, average=None).to(device),
            'f1': MultilabelF1Score(num_labels=self.num_tasks, average=None).to(device),
            # 'precision': MultilabelPrecision(num_labels=self.num_tasks, average=None).to(device),
            # 'recall': MultilabelRecall(num_labels=self.num_tasks, average=None).to(device),
            'auroc': MultilabelAUROC(num_labels=self.num_tasks, average=None).to(device),
            # 'auprc': MultilabelAveragePrecision(num_labels=self.num_tasks, average=None).to(device),
        }
        self._metric_has_data = {name: False for name, metric in self.metrics.items()}
    
    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # logits -> probs via sigmoid
        probs = torch.sigmoid(pred)
        label = label.long()
        
        if name in ['accuracy', 'f1', 'precision', 'recall']:
            pred = (probs > 0.5).long()  # thresholded preds
        elif name in ['auroc', 'auprc']:
            pred = probs

        return pred, label

    def compute(self) -> dict:
        metrics_dict = {}
        for metric_name, metric in self.metrics.items():
            task_values = metric.compute()
            
            # mean of the metric across all tasks
            metrics_dict[metric_name] = task_values.mean()
            
            # the metric for each task
            for i, val in enumerate(task_values):
                task_tag = self.task_names[i]
                metrics_dict[f'{metric_name}_{task_tag}'] = val
                
        return metrics_dict

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
        self._metric_has_data = {name: False for name, metric in self.metrics.items()}
    
    def _prepare_inputs(self, name: str, pred: torch.Tensor, label: torch.Tensor, mask: Optional[torch.Tensor] = None):
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


