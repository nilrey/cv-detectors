from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    """Резолвит 'auto' и проверяет доступность CUDA.

    - 'auto' -> 'cuda' если доступна, иначе 'cpu'.
    - 'cuda*' при отсутствии CUDA -> 'cpu' с предупреждением.
    - остальное возвращается как есть.
    """
    import torch

    if requested == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("device auto resolved to %s", resolved)
        return resolved
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("cuda requested (%s) but unavailable, falling back to cpu", requested)
        return "cpu"
    return requested