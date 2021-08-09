# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.
from typing import Any, Callable, Tuple, Optional


class YOLORuntimeBase:
    """
    PyTorch Lightning wrapper of `YOLO`
    """
    def __init__(
        self,
        engine,
        size: Tuple[int, int] = (640, 640),
        **kwargs: Any,
    ):
        """
        Args:
            arch: architecture
            pretrained: if true, returns a model pre-trained on COCO train2017
            num_classes: number of detection classes (doesn't including background)
        """
        self.engine = engine
        self.transform = Transform(min(size), max(size), fixed_size=size)

    def predict(
        self,
        x: Any,
        image_loader: Optional[Callable] = None,
    ):
        """
        Predict function for raw data or processed data
        Args:
            x: Input to predict. Can be raw data or processed data.
            image_loader: Utility function to convert raw data to Tensor.

        Returns:
            The post-processed model predictions.
        """
        image_loader = image_loader or self.default_loader
        images = self.collate_images(x, image_loader)
        outputs = self.forward(images)
        return outputs

    def default_loader(self, img_path: str):
        """
        Default loader of read a image path.

        Args:
            img_path (str): a image path

        Returns:
            Tensor, processed tensor for prediction.
        """
        return read_image(img_path) / 255.

    def collate_images(self, samples: Any, image_loader: Callable):
        """
        Prepare source samples for inference.

        Args:
            samples (Any): samples source, support the following various types:
                - str or List[str]: a image path or list of image paths.
                - Tensor or List[Tensor]: a tensor or list of tensors.

        Returns:
            List[Tensor], The processed image samples.
        """
        p = next(self.parameters())  # for device and type
        if isinstance(samples, Tensor):
            return [samples.to(p.device).type_as(p)]

        if contains_any_tensor(samples):
            return [sample.to(p.device).type_as(p) for sample in samples]

        if isinstance(samples, str):
            samples = [samples]

        if isinstance(samples, (list, tuple)) and all(isinstance(p, str) for p in samples):
            outputs = []
            for sample in samples:
                output = image_loader(sample).to(p.device).type_as(p)
                outputs.append(output)
            return outputs

        raise NotImplementedError(
            f"The type of the sample is {type(samples)}, we currently don't support it now, the "
            "samples should be either a tensor, list of tensors, a image path or list of image paths."
        )
