from typing import Tuple, Optional
from omegaconf import DictConfig
import random
import math

import numpy as np
import torch
import pytorchvideo.transforms as TV
import torchvision.transforms as T
from torchvision.transforms import v2

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class MinMaxNormalization(object):
    def __call__(self, img):
        """Convert image from [0, 255] to [0, 1]."""
        img = img.float()
        return (img - 0) / 255

class ImageNetNormalization:
    def __init__(self):
        self.mean = IMAGENET_DEFAULT_MEAN
        self.std = IMAGENET_DEFAULT_STD
        self.transform = TV.Normalize(mean=self.mean, std=self.std)

    def __call__(self, x):
        """Apply ImageNet normalization to a tensor or video tensor."""
        return self.transform(x)

class EchoAugmentations():
    def __init__(
        self,
        cfg: DictConfig,
        crop_size: int,
        task_type: str
    ):
        super(EchoAugmentations, self).__init__()

        self.cfg = cfg
        self.crop_size = crop_size
        self.task_type = task_type
        self.apply_augmentations = cfg.apply_augmentations

        self.transforms_to_apply = []
        self.transforms_to_apply_for_segmap = []

        self._add_resize_fn()
        self._add_minmax_norm_fn()
        self._add_imagenet_norm_fn()

        self.transform_fn = TV.ApplyTransformToKey(
            key="video",
            transform=T.Compose(self.transforms_to_apply)
        )

        if self.task_type == 'segmentation':
            self.transform_fn_for_segmap = TV.ApplyTransformToKey(
                key="seg",
                transform=T.Compose(self.transforms_to_apply_for_segmap)
            )

        self.augmentations_to_apply = []

        if self.apply_augmentations:
            self._add_random_horizontal_flip_fn()
            self._add_random_vertical_flip_fn()
            # self._add_random_rotate_fn()
            # self._add_random_crop_fn()

            self.augment_fn = TV.ApplyTransformToKey(
                key="video",
                transform=T.Compose(self.augmentations_to_apply)
            )
        
    def _add_resize_fn(self):
        self.transforms_to_apply.append(T.Resize(self.crop_size, interpolation=T.InterpolationMode.BILINEAR))

        if self.task_type == 'segmentation':
            self.transforms_to_apply_for_segmap.append(T.Resize(self.crop_size, interpolation=T.InterpolationMode.NEAREST))
    
    def _add_minmax_norm_fn(self):
        self.transforms_to_apply.append(MinMaxNormalization())

    def _add_imagenet_norm_fn(self):
        self.transforms_to_apply.append(TV.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
    
    def _add_random_horizontal_flip_fn(self):
        self.augmentations_to_apply.append(T.RandomHorizontalFlip(p=self.cfg.augmentations.random_horizontal_flip))
    
    def _add_random_vertical_flip_fn(self):
        self.augmentations_to_apply.append(T.RandomVerticalFlip(p=self.cfg.augmentations.random_vertical_flip))

    def _add_random_rotate_fn(self):
        self.augmentations_to_apply.append(T.RandomRotation(
            degrees=self.cfg.augmentations.random_rotation,
            interpolation=T.InterpolationMode.BILINEAR,
        ))
    
    def _add_random_crop_fn(self):
        self.augmentations_to_apply.append(T.RandomCrop(
            (self.cfg.augmentations.random_crop_size,
            self.cfg.augmentations.random_crop_size)
        ))

    def __call__(self, echo: torch.Tensor, seg: Optional[torch.Tensor] = None, use_augmentations: bool = True) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]: 
        inp = {'video': echo}
        out = self.transform_fn(inp)
        echo = out['video']

        if seg is not None:
            inp.update({'seg': seg})
            out = self.transform_fn_for_segmap(inp)
            seg = out['seg']

        if use_augmentations:
            inp = {'video': echo}
            if seg is not None:
                inp.update({'seg': seg})
            out = self.augment_fn(inp)
            echo = out['video']
            if seg is not None:
                seg = out['seg']
        
        if seg is not None:
            return echo, seg
        else:
            return echo

class PanEchoAugmentations():
    def __init__(
        self,
        cfg: DictConfig,
        crop_size: int,
        task_type: str
    ):
        super(PanEchoAugmentations, self).__init__()

        self.cfg = cfg
        self.crop_size = crop_size
        self.task_type = task_type
        self.apply_augmentations = cfg.apply_augmentations

        self.transforms_to_apply = []
        self.augmentations_to_apply = []

        if self.apply_augmentations:
            if self.task_type == 'segmentation':
                raise NotImplementedError("Augmentations for segmentation task not implemented yet for PanEchoAugmentations.")
            else:
                self._add_random_zoom_out_fn()
                self._add_random_crop_fn()
                self._add_random_horizontal_flip_fn()
                self._add_random_rotation_fn()

                self.augment_fn = v2.Compose(
                    transforms=self.augmentations_to_apply
                )

                self._add_to_dtype_fn()
                self._add_imagenet_norm_fn()
            
                self.transform_fn = v2.Compose(
                    transforms=self.transforms_to_apply
                )
                
        else:
            self._add_center_crop_fn()
            self._add_to_dtype_fn()
            self._add_imagenet_norm_fn()
            self.transform_fn = v2.Compose(
                transforms=self.transforms_to_apply
            )
    
    def _add_random_zoom_out_fn(self):
        self.augmentations_to_apply.append(v2.RandomZoomOut(fill=0, side_range=(1., 1.2), p=0.5))
    
    def _add_random_crop_fn(self):
        self.augmentations_to_apply.append(v2.RandomCrop(size=(self.crop_size, self.crop_size)))
    
    def _add_random_horizontal_flip_fn(self):
        self.augmentations_to_apply.append(v2.RandomHorizontalFlip(p=0.5))
    
    def _add_random_rotation_fn(self):
        self.augmentations_to_apply.append(v2.RandomRotation(degrees=(-15, 15)))
    
    def _add_center_crop_fn(self):
        self.transforms_to_apply.append(v2.CenterCrop(size=(self.crop_size, self.crop_size)))
    
    def _add_to_dtype_fn(self):
        self.transforms_to_apply.append(v2.ToDtype(torch.float32, scale=True)) # scale=True scales the values to [0, 1]
    
    def _add_imagenet_norm_fn(self):
        self.transforms_to_apply.append(v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))

    def __call__(self, echo: torch.Tensor, use_augmentations: bool = True) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]: 
        if use_augmentations:
            echo = self.augment_fn(echo)
        
        echo = self.transform_fn(echo)

        return echo

class EchoJEPAAugmentations(object):
    def __init__(
        self,
        cfg: DictConfig,
        crop_size: int,
        task_type: str
    ):
        super(EchoJEPAAugmentations, self).__init__()

        self.cfg = cfg
        self.crop_size = crop_size
        self.task_type = task_type
        self.apply_augmentations = cfg.apply_augmentations

        # keep fixed for now
        self.random_horizontal_flip = True
        self.random_resize_aspect_ratio = (3 / 4, 4 / 3)
        self.random_resize_scale = (0.3, 1.0)
        self.auto_augment = False
        self.motion_shift = False
        self.normalize = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.mean = torch.tensor(self.normalize[0], dtype=torch.float32)
        self.std = torch.tensor(self.normalize[1], dtype=torch.float32)
        if not self.auto_augment:
            # Without auto-augment, PIL and tensor conversions simply scale uint8 space by 255.
            self.mean *= 255.0
            self.std *= 255.0

        self.spatial_transform = (
            random_resized_crop_with_shift if self.motion_shift else random_resized_crop
        )

        self.reprob = 0.0

    def __call__(self, echo: torch.Tensor, use_augmentations: bool = True) -> torch.Tensor:

        if torch.is_tensor(echo):
            echo = echo.to(torch.float32)
        else:
            echo = torch.tensor(echo, dtype=torch.float32)

        echo = echo.permute(3, 0, 1, 2)  # T H W C -> C T H W

        if self.apply_augmentations and use_augmentations:
            echo = self.spatial_transform(
                images=echo,
                target_height=self.crop_size,
                target_width=self.crop_size,
                scale=self.random_resize_scale,
                ratio=self.random_resize_aspect_ratio,
            )
            if self.random_horizontal_flip:
                echo, _ = horizontal_flip(0.5, echo)

        echo = _tensor_normalize_inplace(echo, self.mean, self.std)

        return echo


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if isinstance(mean, list):
        mean = torch.tensor(mean)
    if isinstance(std, list):
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def _tensor_normalize_inplace(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize (with dimensions C, T, H, W).
        mean (tensor): mean value to subtract (in 0 to 255 floats).
        std (tensor): std to divide (in 0 to 255 floats).
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()

    C, T, H, W = tensor.shape
    tensor = tensor.view(C, -1).permute(1, 0)  # Make C the last dimension
    tensor.sub_(mean).div_(std)
    tensor = tensor.permute(1, 0).view(C, T, H, W)  # Put C back in front
    return tensor

def _get_param_spatial_crop(scale, ratio, height, width, num_repeat=10, log_scale=True, switch_hw=False):
    """
    Given scale, ratio, height and width, return sampled coordinates of the videos.
    """
    for _ in range(num_repeat):
        area = height * width
        target_area = random.uniform(*scale) * area
        if log_scale:
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
        else:
            aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if np.random.uniform() < 0.5 and switch_hw:
            w, h = h, w

        if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            return i, j, h, w

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


def random_resized_crop(
    images,
    target_height,
    target_width,
    scale=(0.8, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
):
    """
    Crop the given images to random size and aspect ratio. A crop of random
    size (default: of 0.08 to 1.0) of the original size and a random aspect
    ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This
    crop is finally resized to given size. This is popularly used to train the
    Inception networks.

    Args:
        images: Images to perform resizing and cropping.
        target_height: Desired height after cropping.
        target_width: Desired width after cropping.
        scale: Scale range of Inception-style area based random resizing.
        ratio: Aspect ratio range of Inception-style area based random resizing.
    """

    height = images.shape[2]
    width = images.shape[3]

    i, j, h, w = _get_param_spatial_crop(scale, ratio, height, width)
    cropped = images[:, :, i : i + h, j : j + w]
    return torch.nn.functional.interpolate(
        cropped,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )


def random_resized_crop_with_shift(
    images,
    target_height,
    target_width,
    scale=(0.8, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
):
    """
    This is similar to random_resized_crop. However, it samples two different
    boxes (for cropping) for the first and last frame. It then linearly
    interpolates the two boxes for other frames.

    Args:
        images: Images to perform resizing and cropping.
        target_height: Desired height after cropping.
        target_width: Desired width after cropping.
        scale: Scale range of Inception-style area based random resizing.
        ratio: Aspect ratio range of Inception-style area based random resizing.
    """
    t = images.shape[1]
    height = images.shape[2]
    width = images.shape[3]

    i, j, h, w = _get_param_spatial_crop(scale, ratio, height, width)
    i_, j_, h_, w_ = _get_param_spatial_crop(scale, ratio, height, width)
    i_s = [int(i) for i in torch.linspace(i, i_, steps=t).tolist()]
    j_s = [int(i) for i in torch.linspace(j, j_, steps=t).tolist()]
    h_s = [int(i) for i in torch.linspace(h, h_, steps=t).tolist()]
    w_s = [int(i) for i in torch.linspace(w, w_, steps=t).tolist()]
    out = torch.zeros((3, t, target_height, target_width))
    for ind in range(t):
        out[:, ind : ind + 1, :, :] = torch.nn.functional.interpolate(
            images[
                :,
                ind : ind + 1,
                i_s[ind] : i_s[ind] + h_s[ind],
                j_s[ind] : j_s[ind] + w_s[ind],
            ],
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
    return out

def horizontal_flip(prob, images, boxes=None):
    """
    Perform horizontal flip on the given images and corresponding boxes.
    Args:
        prob (float): probility to flip the images.
        images (tensor): images to perform horizontal flip, the dimension is
            `num frames` x `channel` x `height` x `width`.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        images (tensor): images with dimension of
            `num frames` x `channel` x `height` x `width`.
        flipped_boxes (ndarray or None): the flipped boxes with dimension of
            `num boxes` x 4.
    """
    if boxes is None:
        flipped_boxes = None
    else:
        flipped_boxes = boxes.copy()

    if np.random.uniform() < prob:
        images = images.flip((-1))

        if len(images.shape) == 3:
            width = images.shape[2]
        elif len(images.shape) == 4:
            width = images.shape[3]
        else:
            raise NotImplementedError("Dimension does not supported")
        if boxes is not None:
            flipped_boxes[:, [0, 2]] = width - boxes[:, [2, 0]] - 1

    return images, flipped_boxes
