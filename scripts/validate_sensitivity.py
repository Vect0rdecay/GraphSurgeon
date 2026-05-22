#!/usr/bin/env python3
"""Maintainer-only ONNX input sensitivity probe (CPU, onnxruntime).

Not part of the graph-surgeon CLI. Optional check: does model output change under
small input noise? No gradient computation.

Usage:
  .venv/bin/python scripts/validate_sensitivity.py /path/to/model.onnx
"""

import numpy as np
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Callable, Tuple
from datetime import datetime
import logging

try:
    from graph_surgeon._env import import_onnxruntime

    ort = import_onnxruntime()
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

logger = logging.getLogger(__name__)


@dataclass
class PerturbationResult:
    """Result of a single perturbation test."""
    perturbation_level: float  # SNR (dB) for audio, epsilon for vision
    perturbation_type: str  # "snr_db" or "epsilon"
    output_changed: bool
    original_output_hash: str
    perturbed_output_hash: str
    l2_distance: float
    max_output_diff: float
    
    # Legacy compatibility
    @property
    def snr_db(self) -> float:
        return self.perturbation_level if self.perturbation_type == "snr_db" else 0.0


@dataclass  
class SensitivityReport:
    """Complete sensitivity analysis report."""
    model_name: str
    model_path: str
    
    # Sensitivity metrics (domain-agnostic)
    sensitivity_threshold: float  # Perturbation level at which output first changes
    sensitivity_threshold_type: str  # "snr_db" or "epsilon"
    sensitivity_score: float  # 0-100, higher = more sensitive/vulnerable
    model_domain: str  # "audio" or "vision"
    
    # Statistical confidence (new: addresses review feedback)
    sensitivity_score_ci_lower: Optional[float] = None  # 95% CI lower bound
    sensitivity_score_ci_upper: Optional[float] = None  # 95% CI upper bound
    sensitivity_score_std: Optional[float] = None  # Standard deviation across samples
    
    # Optional comparison to an external structural report JSON
    external_risk_score: Optional[float] = None
    correlation_valid: bool = False
    
    # Test details
    perturbation_levels_tested: List[float] = None
    results_by_level: Dict[float, PerturbationResult] = None
    num_samples: int = 0
    
    # Metadata
    timestamp: str = ""
    
    # Legacy compatibility
    @property
    def sensitivity_threshold_db(self) -> float:
        return self.sensitivity_threshold if self.sensitivity_threshold_type == "snr_db" else 0.0
    
    @property
    def snr_levels_tested(self) -> List[float]:
        return self.perturbation_levels_tested if self.sensitivity_threshold_type == "snr_db" else []
    
    def to_dict(self) -> Dict:
        result = {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "model_domain": self.model_domain,
            "sensitivity_threshold": self.sensitivity_threshold,
            "sensitivity_threshold_type": self.sensitivity_threshold_type,
            "sensitivity_score": self.sensitivity_score,
            "external_risk_score": self.external_risk_score,
            "correlation_valid": self.correlation_valid,
            "perturbation_levels_tested": self.perturbation_levels_tested,
            "num_samples": self.num_samples,
            "timestamp": self.timestamp,
        }
        # Add confidence interval if computed
        if self.sensitivity_score_ci_lower is not None:
            result["confidence_interval"] = {
                "lower": self.sensitivity_score_ci_lower,
                "upper": self.sensitivity_score_ci_upper,
                "std": self.sensitivity_score_std,
            }
        return result


class FastValidator:
    """
    Fast perturbation sensitivity validator.
    
    Tests if a model is sensitive to noise perturbations at domain-appropriate levels:
    - Audio: SNR levels (dB) - higher SNR = smaller perturbation
    - Vision: Epsilon levels - lower epsilon = smaller perturbation
    
    A model that changes output at small perturbations is more vulnerable.
    """
    
    # Domain-specific perturbation levels
    DEFAULT_SNR_LEVELS = [40.0, 30.0, 20.0, 10.0, 5.0]  # Audio: dB
    DEFAULT_EPSILON_LEVELS = [0.001, 0.005, 0.01, 0.05, 0.1]  # Vision: L∞ bound
    
    # Statistical validity: n=5 gives 95% CI of [15%, 95%] for 60% observed rate
    # n=30 gives 95% CI of [40%, 77%] - much more reliable
    DEFAULT_NUM_SAMPLES = 30
    
    def __init__(
        self,
        snr_levels: List[float] = None,
        epsilon_levels: List[float] = None,
        num_samples: int = None,
        output_dir: str = "validation_results",
    ):
        """
        Args:
            snr_levels: SNR levels to test for audio (dB), default [40, 30, 20, 10, 5]
            epsilon_levels: Epsilon levels to test for vision, default [0.001, 0.005, 0.01, 0.05, 0.1]
            num_samples: Number of random samples to test per level (default: 30 for statistical validity)
            output_dir: Directory to save results
        """
        self.snr_levels = snr_levels or self.DEFAULT_SNR_LEVELS
        self.epsilon_levels = epsilon_levels or self.DEFAULT_EPSILON_LEVELS
        self.num_samples = num_samples if num_samples is not None else self.DEFAULT_NUM_SAMPLES
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def validate_onnx_model(
        self,
        model_path: str,
        external_report: Optional[Dict] = None,
        input_shape: Optional[Tuple] = None,
    ) -> SensitivityReport:
        """
        Validate an ONNX model's perturbation sensitivity.
        
        Args:
            model_path: Path to ONNX model
            external_report: Optional external JSON report for comparison
            input_shape: Input shape (auto-detected if not provided)
        
        Returns:
            SensitivityReport with results
        """
        if not HAS_ONNX:
            raise RuntimeError("onnxruntime not installed")
        
        print(f"Loading model: {model_path}")
        session = ort.InferenceSession(model_path)
        
        # Get input info
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        
        if input_shape is None:
            # Try to get shape from model
            shape = input_info.shape
            # Replace dynamic dims with reasonable values for vision models
            # Typical shapes: [batch, channels, height, width] or [batch, height, width, channels]
            input_shape = []
            for i, dim in enumerate(shape):
                if isinstance(dim, int) and dim > 0:
                    input_shape.append(dim)
                elif i == 0:  # batch dimension
                    input_shape.append(1)
                elif i == 1:  # channels (NCHW) or height (NHWC)
                    # Check if this looks like NCHW (channels first)
                    if len(shape) == 4:
                        input_shape.append(3)  # RGB channels
                    else:
                        input_shape.append(224)
                else:  # height, width
                    input_shape.append(224)
            input_shape = tuple(input_shape)
        
        print(f"Input shape: {input_shape}")
        
        # Create model function
        output_name = session.get_outputs()[0].name
        
        def model_fn(x: np.ndarray) -> np.ndarray:
            return session.run([output_name], {input_name: x.astype(np.float32)})[0]
        
        # Run sensitivity test (vision models use epsilon)
        return self._run_sensitivity_test(
            model_fn=model_fn,
            input_shape=input_shape,
            model_name=os.path.basename(model_path),
            model_path=model_path,
            external_report=external_report,
            model_domain="vision",
        )
    
    def validate_audio_model(
        self,
        model_path: str,
        external_report: Optional[Dict] = None,
        sample_rate: int = 16000,
        duration: float = 3.0,
    ) -> SensitivityReport:
        """
        Validate an audio model's perturbation sensitivity.
        
        Args:
            model_path: Path to ONNX audio model
            external_report: Optional external structural report
            sample_rate: Audio sample rate
            duration: Audio duration in seconds
        
        Returns:
            SensitivityReport
        """
        if not HAS_ONNX:
            raise RuntimeError("onnxruntime not installed")
        
        print(f"Loading audio model: {model_path}")
        session = ort.InferenceSession(model_path)
        
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape
        
        # Determine input shape for audio
        # Common shapes: [batch, channels, time] or [batch, mel_bins, time]
        if input_shape:
            resolved_shape = []
            for i, dim in enumerate(input_shape):
                if isinstance(dim, int) and dim > 0:
                    resolved_shape.append(dim)
                elif i == 0:  # batch
                    resolved_shape.append(1)
                elif i == 1:  # channels or mel bins
                    resolved_shape.append(80)  # typical mel bins
                else:  # time
                    resolved_shape.append(int(sample_rate * duration / 160))  # ~100 fps
            input_shape = tuple(resolved_shape)
        else:
            input_shape = (1, 80, int(sample_rate * duration / 160))
        
        print(f"Audio input shape: {input_shape}")
        
        output_name = session.get_outputs()[0].name
        
        def model_fn(x: np.ndarray) -> np.ndarray:
            return session.run([output_name], {input_name: x.astype(np.float32)})[0]
        
        # Run sensitivity test (audio models use SNR)
        return self._run_sensitivity_test(
            model_fn=model_fn,
            input_shape=input_shape,
            model_name=os.path.basename(model_path),
            model_path=model_path,
            external_report=external_report,
            model_domain="audio",
        )
    
    def _run_sensitivity_test(
        self,
        model_fn: Callable,
        input_shape: Tuple,
        model_name: str,
        model_path: str,
        external_report: Optional[Dict],
        model_domain: str = "audio",  # "audio" or "vision"
    ) -> SensitivityReport:
        """Run the actual sensitivity test with domain-appropriate metrics."""
        
        # Select perturbation levels based on domain
        if model_domain == "vision":
            perturbation_levels = self.epsilon_levels
            perturbation_type = "epsilon"
            unit = "ε"
        else:  # audio
            perturbation_levels = self.snr_levels
            perturbation_type = "snr_db"
            unit = "dB"
        
        print(f"\nRunning sensitivity test on {model_name}")
        print(f"Domain: {model_domain.upper()}")
        if model_domain == "vision":
            print(f"Testing epsilon levels: {perturbation_levels}")
        else:
            print(f"Testing SNR levels: {perturbation_levels} dB")
        print(f"Samples per level: {self.num_samples}")
        print("-" * 50)
        
        results_by_level = {}
        
        # Initialize threshold based on domain
        if model_domain == "vision":
            sensitivity_threshold = perturbation_levels[-1]  # Start with largest epsilon (most noise)
        else:
            sensitivity_threshold = perturbation_levels[-1]  # Start with lowest SNR (most noise)
        
        for level in perturbation_levels:
            changes_detected = 0
            total_max_diff = 0.0
            
            for sample_idx in range(self.num_samples):
                # Generate random input (simulating real data)
                if sample_idx % 3 == 0:
                    x_clean = np.random.randn(*input_shape).astype(np.float32) * 0.5
                elif sample_idx % 3 == 1:
                    x_clean = np.random.uniform(-1, 1, input_shape).astype(np.float32)
                else:
                    x_clean = np.zeros(input_shape, dtype=np.float32)
                    mask = np.random.random(input_shape) > 0.7
                    x_clean[mask] = np.random.randn(mask.sum()).astype(np.float32)
                
                # Get clean output
                try:
                    y_clean = model_fn(x_clean)
                except Exception as e:
                    print(f"  Error on clean input: {e}")
                    continue
                
                # Add noise based on domain
                if model_domain == "vision":
                    x_noisy = self._add_noise_epsilon(x_clean, level)
                else:
                    x_noisy = self._add_noise_snr(x_clean, level)
                
                # Get noisy output
                try:
                    y_noisy = model_fn(x_noisy)
                except Exception as e:
                    print(f"  Error on noisy input: {e}")
                    continue
                
                # Check if output changed (using adaptive detection)
                changed, change_metrics = self._detect_output_change(y_clean, y_noisy)
                
                # Also compute normalized metrics for reporting
                norm_metrics = self._compute_normalized_change(y_clean, y_noisy)
                
                max_diff = norm_metrics["max_change"]
                total_max_diff += max_diff
                
                if changed:
                    changes_detected += 1
            
            change_rate = changes_detected / self.num_samples
            avg_max_diff = total_max_diff / self.num_samples
            
            if model_domain == "vision":
                print(f"  ε={level:6.4f}: {change_rate*100:5.1f}% outputs changed, "
                      f"avg max diff: {avg_max_diff:.4f}")
            else:
                print(f"  SNR {level:5.1f} dB: {change_rate*100:5.1f}% outputs changed, "
                      f"avg max diff: {avg_max_diff:.4f}")
            
            results_by_level[level] = PerturbationResult(
                perturbation_level=level,
                perturbation_type=perturbation_type,
                output_changed=change_rate > 0.5,
                original_output_hash="",
                perturbed_output_hash="",
                l2_distance=0.0,
                max_output_diff=avg_max_diff,
            )
            
            # Update sensitivity threshold
            if change_rate > 0.5:
                if model_domain == "vision":
                    # For vision: smaller epsilon = more sensitive
                    if level < sensitivity_threshold:
                        sensitivity_threshold = level
                else:
                    # For audio: higher SNR = more sensitive
                    if level > sensitivity_threshold:
                        sensitivity_threshold = level
        
        # Calculate sensitivity score (0-100) - domain-specific
        if model_domain == "vision":
            # Vision: ε=0.001 -> very sensitive (score ~90)
            #         ε=0.1   -> robust (score ~0)
            sensitivity_score = min(100, max(0, (1 - sensitivity_threshold / 0.1) * 100))
        else:
            # Audio: 40 dB -> very sensitive (score ~87.5)
            #        5 dB  -> robust (score ~0)
            sensitivity_score = min(100, max(0, (sensitivity_threshold - 5) * 2.5))
        
        print("-" * 50)
        if model_domain == "vision":
            print(f"Sensitivity threshold: ε={sensitivity_threshold}")
        else:
            print(f"Sensitivity threshold: {sensitivity_threshold} dB")
        print(f"Sensitivity score: {sensitivity_score:.1f}/100")
        
        # Compare with external report if available
        external_risk = None
        correlation_valid = False
        if external_report:
            external_risk = external_report.get("overall_risk_score", 
                          external_report.get("adversarial_perturbation_risk", 0))
            
            # Check if external risk score correlates with sensitivity
            external_high = external_risk > 50
            sensitivity_high = sensitivity_score > 50
            correlation_valid = (external_high == sensitivity_high)
            
            print(f"\nExternal risk score: {external_risk:.1f}/100")
            print(f"Correlation: {'aligned' if correlation_valid else 'mismatch'}")
        
        # Create report with domain-appropriate fields
        report = SensitivityReport(
            model_name=model_name,
            model_path=model_path,
            sensitivity_threshold=sensitivity_threshold,
            sensitivity_threshold_type=perturbation_type,
            sensitivity_score=sensitivity_score,
            model_domain=model_domain,
            external_risk_score=external_risk,
            correlation_valid=correlation_valid,
            perturbation_levels_tested=perturbation_levels,
            results_by_level=results_by_level,
            num_samples=self.num_samples,
            timestamp=datetime.now().isoformat(),
        )
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _add_noise_snr(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Add Gaussian noise to achieve target SNR (for audio models)."""
        signal_power = np.mean(signal ** 2)
        
        # SNR = 10 * log10(signal_power / noise_power)
        # noise_power = signal_power / 10^(SNR/10)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise_std = np.sqrt(noise_power)
        
        noise = np.random.randn(*signal.shape).astype(signal.dtype) * noise_std
        return signal + noise
    
    def _add_noise_epsilon(self, signal: np.ndarray, epsilon: float) -> np.ndarray:
        """Add uniform noise within L∞ bound epsilon (for vision models)."""
        # Uniform noise in [-epsilon, epsilon]
        noise = np.random.uniform(-epsilon, epsilon, signal.shape).astype(signal.dtype)
        return signal + noise
    
    def _infer_output_type(self, output: np.ndarray) -> str:
        """
        Infer output type from output characteristics.
        
        Addresses review feedback: different output types need different change metrics.
        
        Returns one of: "probabilities", "logits", "embeddings", "raw"
        """
        flat = output.flatten()
        
        # Check if looks like probabilities (sums to ~1, all non-negative)
        if np.all(flat >= -1e-6) and 0.95 < np.sum(flat) < 1.05:
            return "probabilities"
        
        # Check if looks like log-probabilities (all negative, exp sums to ~1)
        if np.all(flat <= 0) and 0.95 < np.sum(np.exp(flat)) < 1.05:
            return "log_probabilities"
        
        # Check if looks like logits (reasonable range, can be negative)
        output_range = np.max(flat) - np.min(flat)
        if -50 < np.min(flat) < 50 and output_range < 100:
            # Check dimensionality - low dim likely classification logits
            if output.size < 10000:
                return "logits"
        
        # High-dimensional output likely embeddings
        if output.size > 10000:
            return "embeddings"
        
        return "raw"
    
    def _detect_output_change(
        self, 
        y_clean: np.ndarray, 
        y_noisy: np.ndarray, 
        output_type: str = "auto"
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Detect if output changed using output-type-appropriate metric.
        
        Addresses review feedback: fixed 1% threshold may not be appropriate for all output types.
        
        Args:
            y_clean: Clean output
            y_noisy: Noisy output  
            output_type: One of "probabilities", "logits", "embeddings", "raw", or "auto"
        
        Returns:
            Tuple of (changed: bool, metrics: dict with detailed change info)
        """
        if output_type == "auto":
            output_type = self._infer_output_type(y_clean)
        
        metrics = {
            "output_type": output_type,
            "total_dims": y_clean.size,
        }
        
        if output_type == "probabilities":
            # Use KL divergence for probability distributions
            eps = 1e-10
            y_clean_safe = np.clip(y_clean, eps, 1.0)
            y_noisy_safe = np.clip(y_noisy, eps, 1.0)
            kl_div = np.sum(y_clean_safe * np.log(y_clean_safe / y_noisy_safe))
            
            metrics["kl_divergence"] = float(kl_div)
            metrics["threshold"] = 0.01  # ~1% probability mass shift
            changed = kl_div > 0.01
            
        elif output_type == "log_probabilities":
            # Use KL divergence on exp
            eps = 1e-10
            p_clean = np.exp(y_clean)
            p_noisy = np.exp(y_noisy)
            kl_div = np.sum(p_clean * (y_clean - y_noisy))
            
            metrics["kl_divergence"] = float(kl_div)
            metrics["threshold"] = 0.01
            changed = kl_div > 0.01
            
        elif output_type == "embeddings":
            # Use cosine distance for embeddings
            flat_clean = y_clean.flatten()
            flat_noisy = y_noisy.flatten()
            norm_clean = np.linalg.norm(flat_clean)
            norm_noisy = np.linalg.norm(flat_noisy)
            
            if norm_clean < 1e-10 or norm_noisy < 1e-10:
                cos_sim = 0.0
            else:
                cos_sim = np.dot(flat_clean, flat_noisy) / (norm_clean * norm_noisy)
            
            metrics["cosine_similarity"] = float(cos_sim)
            metrics["threshold"] = 0.99  # 1% angular difference
            changed = cos_sim < 0.99
            
        else:  # logits or raw outputs
            # Use relative L∞ distance (original method)
            output_range = np.max(y_clean) - np.min(y_clean) + 1e-8
            max_diff = np.max(np.abs(y_clean - y_noisy))
            relative_diff = max_diff / output_range
            
            metrics["max_diff"] = float(max_diff)
            metrics["output_range"] = float(output_range)
            metrics["relative_diff"] = float(relative_diff)
            metrics["threshold"] = 0.01  # 1% of range
            changed = relative_diff > 0.01
        
        return changed, metrics
    
    def _compute_normalized_change(
        self, 
        y_clean: np.ndarray, 
        y_noisy: np.ndarray,
        threshold_fraction: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Compute change metrics normalized by output dimensionality.
        
        Addresses review feedback: high-dimensional encoders appear more sensitive
        because any of N outputs changing triggers "changed" flag.
        
        Args:
            y_clean: Clean output
            y_noisy: Noisy output
            threshold_fraction: Fraction of output range to consider "changed"
        
        Returns:
            Dict with normalized metrics
        """
        total_dims = y_clean.size
        output_range = np.max(y_clean) - np.min(y_clean) + 1e-8
        threshold = threshold_fraction * output_range
        
        # Count dimensions that changed
        abs_diff = np.abs(y_clean - y_noisy)
        changed_dims = np.sum(abs_diff > threshold)
        
        # Normalized change fraction
        change_fraction = changed_dims / total_dims
        
        # Mean and max change
        mean_change = np.mean(abs_diff)
        max_change = np.max(abs_diff)
        
        # L2 distance (normalized by dim)
        l2_dist = np.linalg.norm(y_clean - y_noisy)
        l2_per_dim = l2_dist / np.sqrt(total_dims)
        
        return {
            "changed_dims": int(changed_dims),
            "total_dims": total_dims,
            "change_fraction": float(change_fraction),  # 0.01 = 1% of dims changed
            "mean_change": float(mean_change),
            "max_change": float(max_change),
            "l2_distance": float(l2_dist),
            "l2_per_dim": float(l2_per_dim),
            "output_range": float(output_range),
        }
    
    def compute_sensitivity_with_ci(
        self, 
        model_path: str, 
        n_bootstrap: int = 10,
        confidence: float = 0.95,
    ) -> Dict[str, float]:
        """
        Compute sensitivity score with bootstrap confidence interval.
        
        Addresses review feedback: need confidence intervals for statistical validity.
        
        Args:
            model_path: Path to ONNX model
            n_bootstrap: Number of bootstrap trials (default: 10)
            confidence: Confidence level (default: 0.95 for 95% CI)
        
        Returns:
            Dict with score, ci_lower, ci_upper, std
        """
        scores = []
        
        # Temporarily reduce samples per trial for bootstrap efficiency
        original_samples = self.num_samples
        bootstrap_samples = max(10, self.num_samples // 3)  # Use 1/3 samples per bootstrap
        self.num_samples = bootstrap_samples
        
        for trial in range(n_bootstrap):
            try:
                report = self.validate_audio_model(model_path) if self._is_audio_model(model_path) \
                         else self.validate_onnx_model(model_path)
                scores.append(report.sensitivity_score)
            except Exception as e:
                logger.warning(f"Bootstrap trial {trial} failed: {e}")
        
        # Restore original sample count
        self.num_samples = original_samples
        
        if len(scores) < 3:
            return {
                'score': np.mean(scores) if scores else 0.0,
                'ci_lower': 0.0,
                'ci_upper': 100.0,
                'std': 0.0,
                'n_trials': len(scores),
            }
        
        # Compute percentile-based CI
        alpha = (1 - confidence) / 2
        ci_lower = np.percentile(scores, alpha * 100)
        ci_upper = np.percentile(scores, (1 - alpha) * 100)
        
        return {
            'score': np.mean(scores),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': np.std(scores),
            'n_trials': len(scores),
        }
    
    def _is_audio_model(self, model_path: str) -> bool:
        """Heuristic to detect if model is audio-based."""
        path_lower = model_path.lower()
        audio_keywords = ['whisper', 'wav2vec', 'hubert', 'audio', 'speech', 'encoder_model']
        return any(kw in path_lower for kw in audio_keywords)
    
    def _save_report(self, report: SensitivityReport):
        """Save report to JSON file."""
        filename = f"{report.model_name}_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        print(f"\nReport saved to: {filepath}")


def quick_validate(model_path: str, external_report_path: Optional[str] = None) -> SensitivityReport:
    """
    Quick validation function for command-line use.
    
    Args:
        model_path: Path to ONNX model
        external_report_path: Optional path to external JSON report
    
    Returns:
        SensitivityReport
    """
    external_report = None
    if external_report_path and os.path.exists(external_report_path):
        with open(external_report_path) as f:
            external_report = json.load(f)
    
    validator = FastValidator(num_samples=5)
    
    # Detect if audio model based on path
    if "whisper" in model_path.lower() or "wav2vec" in model_path.lower() or "audio" in model_path.lower():
        return validator.validate_audio_model(model_path, external_report)
    else:
        return validator.validate_onnx_model(model_path, external_report)
