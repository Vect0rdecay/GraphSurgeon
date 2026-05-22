"""
ONNX weight distribution analysis for reverse engineering.

This module analyzes ONNX model weight distributions to detect whether
a model was likely adversarially trained. Based on empirical validation
showing that adversarially-trained models have significantly higher
weight kurtosis than standard-trained models.

Key Finding (RobustBench validation, n=12):
- Standard-trained: kurtosis ~4
- Adversarially-trained: kurtosis 9-17
- Classification threshold: 6-8 achieves ~100% accuracy

Reference: Nayebi & Ganguli (2017) "Biologically inspired protection 
of deep networks from adversarial attacks" - demonstrated kurtotic
weight distributions provide adversarial robustness.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum
import numpy as np


class TrainingType(Enum):
    """Detected training methodology."""
    STANDARD = "standard"  # Likely no adversarial training
    UNCERTAIN = "uncertain"  # Borderline kurtosis
    ADVERSARIAL = "adversarial"  # Likely adversarially trained


@dataclass
class WeightAnalysisResult:
    """Results from weight distribution analysis."""
    
    # Core statistics
    avg_kurtosis: float
    std_kurtosis: float
    min_kurtosis: float
    max_kurtosis: float
    num_weight_tensors: int
    total_parameters: int
    
    # Classification
    detected_training: TrainingType
    confidence: float  # 0-1
    
    # Thresholds used
    low_threshold: float = 6.0
    high_threshold: float = 8.0
    
    def __post_init__(self):
        """Compute training classification from kurtosis."""
        if self.avg_kurtosis < self.low_threshold:
            self.detected_training = TrainingType.STANDARD
            # Distance from threshold determines confidence
            self.confidence = min(1.0, (self.low_threshold - self.avg_kurtosis) / 3.0)
        elif self.avg_kurtosis > self.high_threshold:
            self.detected_training = TrainingType.ADVERSARIAL
            self.confidence = min(1.0, (self.avg_kurtosis - self.high_threshold) / 5.0)
        else:
            self.detected_training = TrainingType.UNCERTAIN
            # Confidence is low in the uncertain zone
            self.confidence = 0.3
    
    @property
    def is_likely_robust(self) -> bool:
        """Whether model shows signs of adversarial training."""
        return self.detected_training == TrainingType.ADVERSARIAL
    
    @property
    def is_likely_vulnerable(self) -> bool:
        """Whether model shows signs of standard training (vulnerable)."""
        return self.detected_training == TrainingType.STANDARD
    
    def summary(self) -> str:
        """Human-readable summary."""
        status = {
            TrainingType.STANDARD: "LIKELY VULNERABLE (no AT detected)",
            TrainingType.UNCERTAIN: "UNCERTAIN (borderline kurtosis)",
            TrainingType.ADVERSARIAL: "LIKELY HARDENED (AT detected)",
        }
        return (
            f"Weight Analysis: {status[self.detected_training]}\n"
            f"  Avg Kurtosis: {self.avg_kurtosis:.2f} "
            f"(threshold: <{self.low_threshold} vulnerable, >{self.high_threshold} hardened)\n"
            f"  Confidence: {self.confidence:.0%}\n"
            f"  Tensors analyzed: {self.num_weight_tensors}"
        )


def analyze_onnx_weights(model_path: str, 
                         min_tensor_size: int = 100) -> WeightAnalysisResult:
    """
    Analyze weight distributions in an ONNX model.
    
    Args:
        model_path: Path to ONNX file
        min_tensor_size: Minimum tensor size to include in analysis
        
    Returns:
        WeightAnalysisResult with kurtosis statistics and AT detection
    """
    import onnx
    
    model = onnx.load(model_path)
    
    kurtosis_values = []
    total_params = 0
    
    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        total_params += arr.size
        
        if arr.size >= min_tensor_size:
            # Compute excess kurtosis (normal distribution = 3, excess = 0)
            mean = np.mean(arr)
            std = np.std(arr)
            if std > 1e-10:
                kurtosis = ((arr - mean) ** 4).mean() / (std ** 4)
                kurtosis_values.append(kurtosis)
    
    if not kurtosis_values:
        # No substantial weight tensors found
        return WeightAnalysisResult(
            avg_kurtosis=3.0,  # Normal distribution default
            std_kurtosis=0.0,
            min_kurtosis=3.0,
            max_kurtosis=3.0,
            num_weight_tensors=0,
            total_parameters=total_params,
            detected_training=TrainingType.UNCERTAIN,
            confidence=0.0,
        )
    
    result = WeightAnalysisResult(
        avg_kurtosis=float(np.mean(kurtosis_values)),
        std_kurtosis=float(np.std(kurtosis_values)),
        min_kurtosis=float(np.min(kurtosis_values)),
        max_kurtosis=float(np.max(kurtosis_values)),
        num_weight_tensors=len(kurtosis_values),
        total_parameters=total_params,
        detected_training=TrainingType.STANDARD,  # Will be updated in __post_init__
        confidence=0.0,  # Will be updated in __post_init__
    )
    
    # Trigger classification
    result.__post_init__()
    
    return result
