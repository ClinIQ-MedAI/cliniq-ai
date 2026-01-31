from __future__ import annotations

from ultralytics import YOLO

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_yolo(weights: str) -> YOLO:
    """
    Returns an Ultralytics YOLO model ready for .train/.val/.predict
    weights can be: 'yolov8x.pt' or path to best.pt
    """
    return YOLO(weights)


def build_frcnn(num_classes_with_bg: int, use_v2: bool = True):
    """
    TorchVision Faster R-CNN. num_classes_with_bg = K + 1 (background class 0)
    """
    if use_v2 and hasattr(torchvision.models.detection, "fasterrcnn_resnet50_fpn_v2"):
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
    else:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
    return model


# Registry of available detection models
DETECTION_MODELS = {
    'yolo': build_yolo,
    'fasterrcnn': build_frcnn,
}


def build_detection_model(model_name: str, **kwargs):
    """
    Factory function to build detection models by name.
    
    Args:
        model_name: Name of the model ('yolo' or 'fasterrcnn')
        **kwargs: Model-specific arguments
        
    Returns:
        Detection model instance
    """
    if model_name not in DETECTION_MODELS:
        raise ValueError(f"Unknown detection model: {model_name}. Available: {list(DETECTION_MODELS.keys())}")
    
    return DETECTION_MODELS[model_name](**kwargs)
