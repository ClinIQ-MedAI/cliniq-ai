"""
API module for oral disease classification.
"""

from .inference import OralClassifier, get_classifier, CLASS_NAMES
from .gradcam import GradCAM, GradCAMPlusPlus, get_gradcam_for_convnext

__all__ = [
    'OralClassifier',
    'get_classifier',
    'CLASS_NAMES',
    'GradCAM',
    'GradCAMPlusPlus',
    'get_gradcam_for_convnext'
]
