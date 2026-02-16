"""
EHR-JEPA Benchmark Suite

PhD-Level Evaluation Framework for EHR Foundation Models

Modules:
- preprocess_benchmark: Extract 300+ features from MIMIC-IV
- evaluate_benchmark: Run comprehensive evaluation
- train_baselines: Train neural network baselines
- ablation_study: Run ablation experiments
- generate_paper_artifacts: Create publication figures/tables
"""

from .evaluate_benchmark import BenchmarkEvaluator, BENCHMARK_TASKS
from .ablation_study import AblationStudy

__all__ = [
    'BenchmarkEvaluator',
    'AblationStudy', 
    'BENCHMARK_TASKS'
]
