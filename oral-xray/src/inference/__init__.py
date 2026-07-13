"""
Inference module for dental X-ray analysis.
"""

from .pipeline import DentalInferencePipeline, Detection, InferenceResult

__all__ = [
    'DentalInferencePipeline',
    'Detection', 
    'InferenceResult'
]
