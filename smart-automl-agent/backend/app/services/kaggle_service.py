"""
Kaggle Cloud GPU training — STUB.

In the real production path this would:
  1. Build a notebook payload (training script + dataset) using `kaggle.api`.
  2. Push it to Kaggle with GPU enabled.
  3. Poll for completion, stream logs back over WebSocket.
  4. Download the resulting model artifact.

For the graduation demo we run training locally on CPU. This file defines
the interface so the local path and the cloud path are interchangeable.

To wire this up:
  - Put your kaggle.json in ~/.kaggle/  (chmod 600)
  - `pip install kaggle`
  - Implement `dispatch()` and `poll()` below.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class KaggleGPUService:
    def __init__(self) -> None:
        self.enabled = False  # flip to True once `dispatch` is implemented

    def dispatch(
        self,
        dataset_path: str,
        training_script: str,
        gpu_type: str = "T4",
    ) -> str:
        """Submit a training job. Returns a job_id."""
        if not self.enabled:
            raise NotImplementedError(
                "Kaggle GPU dispatch is stubbed for the demo. "
                "See app/services/kaggle_service.py to enable."
            )

    def poll(self, job_id: str) -> dict[str, Any]:
        """Return {status, logs, artifact_url}."""
        if not self.enabled:
            raise NotImplementedError("Kaggle GPU polling is stubbed for the demo.")


kaggle_gpu = KaggleGPUService()
