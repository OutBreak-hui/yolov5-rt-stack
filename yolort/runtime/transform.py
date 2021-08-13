# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from typing import Dict, Optional, List, Tuple


class YOLORTTransform:
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """
    def __init__(
        self,
        min_size: int,
        max_size: int,
        fixed_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Note: When ``fixed_size`` is set, the ``min_size`` and ``max_size`` won't take effect.
        """
        super().__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.fixed_size = fixed_size

    def __call__(self, images):
        device = images[0].device
        images = [img for img in images]

        for i in range(len(images)):
            image = images[i]

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 f"of shape [C, H, W], got {image.shape}")

            image = self.resize(image)
            images[i] = image

        image_sizes = [img.shape[-2:] for img in images]
        images = nested_tensor_from_tensor_list(images)
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = NestedTensor(images, image_sizes_list)

        return image_list

    def resize(self, image):

        h, w = image.shape[-2:]

        # FIXME assume for now that testing uses the largest scale
        size = float(self.min_size[-1])

        image, target = _resize_image_and_masks(image, size, float(self.max_size), self.fixed_size, target)

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = normalize_boxes(bbox, (h, w))
        target["boxes"] = bbox

        return image, target

    def postprocess(
        self,
        result,
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ):

        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes

        return result
