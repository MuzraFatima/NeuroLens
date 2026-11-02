"""Shared utility helpers for TopoCircuit-SAE.

Provides:
- ``seed_everything``: Set NumPy global random seed.
- ``cosine_similarity_matrix``: Pairwise column cosine similarities.
- ``topk_mask``: Boolean mask for the top-k entries per row.
- ``format_percent``: Human-readable percentage formatter.
- ``load_yaml_config``: Load and validate a YAML config file.
- ``ensure_dir``: Create a directory path if it does not exist.
"""

from __future__ import annotations

import os
from typing import Dict, Union

import numpy as np
import yaml


def seed_everything(seed: int = 42) -> None:
    """Set the NumPy global random seed for reproducibility."""
    np.random.seed(seed)


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return the (m, n) pairwise cosine similarity matrix between columns of A and B.

    Args:
        A: Array of shape ``(d, m)`` with unit-norm or arbitrary-norm columns.
        B: Array of shape ``(d, n)`` with unit-norm or arbitrary-norm columns.

    Returns:
        Matrix of shape ``(m, n)`` where entry ``[i, j] = cos(A[:, i], B[:, j])``.
    """
    A_norm = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    return A_norm.T @ B_norm


def topk_mask(x: np.ndarray, k: int) -> np.ndarray:
    """Return a boolean mask keeping the top-k entries per row of ``x``.

    Args:
        x: 2-D array of shape ``(B, n)``.
        k: Number of entries to keep per row.

    Returns:
        Boolean array of shape ``(B, n)`` with exactly ``min(k, n)`` True per row.
    """
    k_eff = min(max(1, k), x.shape[1])
    mask = np.zeros_like(x, dtype=bool)
    idx = np.argpartition(x, -k_eff, axis=1)[:, -k_eff:]
    np.put_along_axis(mask, idx, True, axis=1)
    return mask


def format_percent(value: float, decimals: int = 2) -> str:
    """Format a float in ``[0, 1]`` as a percentage string.

    Args:
        value: Fraction in the range ``[0, 1]``.
        decimals: Number of decimal places (default 2).

    Returns:
        String such as ``"6.71%"``.
    """
    return f"{value * 100:.{decimals}f}%"


def load_yaml_config(path: str) -> Dict[str, object]:
    """Load a YAML configuration file and return its contents as a dict.

    Args:
        path: Absolute or relative path to the ``.yaml`` / ``.yml`` file.

    Returns:
        Parsed YAML as a nested dictionary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file does not parse to a mapping.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at top level, got {type(data).__name__}")
    return data  # type: ignore[return-value]


def ensure_dir(path: Union[str, os.PathLike]) -> str:
    """Create *path* (and any missing parents) if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved absolute path string.
    """
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def validate_preregistration_config(cfg: Dict[str, object]) -> bool:
    """Validates that a configuration dictionary contains the mandatory pre-registration schema.

    Args:
        cfg: Parsed dictionary from `preregistration_v1.yaml`.

    Returns:
        True if all required keys and structures are present.

    Raises:
        KeyError: If any required section or field is missing.
        TypeError: If fields fail basic type checks.
    """
    required_sections = ["project", "hypotheses", "hyperparameters"]
    for sec in required_sections:
        if sec not in cfg or not isinstance(cfg[sec], dict):
            raise KeyError(f"Missing required configuration section: '{sec}'")

    project = cfg["project"]
    for field in ["name", "version", "owner", "preregistration_tag"]:
        if field not in project or not isinstance(project[field], str):
            raise KeyError(f"Missing or invalid project field: '{field}'")

    hypotheses = cfg["hypotheses"]
    for h_id in [
        "H1_multi_parent_scaling_and_order",
        "H2_mpabs_reduction",
        "H3_edge_validity_and_stability",
        "H4_circuit_compactness",
    ]:
        if h_id not in hypotheses or not isinstance(hypotheses[h_id], dict):
            raise KeyError(f"Missing required hypothesis section: '{h_id}'")

    hyperparameters = cfg["hyperparameters"]
    if "main_llm_setting" not in hyperparameters:
        raise KeyError("Missing 'main_llm_setting' in hyperparameters")

    return True


__all__ = [
    "seed_everything",
    "cosine_similarity_matrix",
    "topk_mask",
    "format_percent",
    "load_yaml_config",
    "ensure_dir",
    "validate_preregistration_config",
]
