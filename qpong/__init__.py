"""Q-PONG: Quantum-Physics-Informed Operations and Negotiated Governance.

Hybrid quantum-classical pipeline for adaptive unit commitment and risk-aware
smart-grid operation:

    Stage I   context-adaptive fuzzy Best-Worst weighting     (qpong.stage1_fbwm)
    Stage II  VQE commitment-candidate generation              (qpong.stage2_quantum)
    Stage III PINN prediction + independent AC verification    (qpong.stage3_pinn)
    Stage IV  Gaussian-fuzzy Shapley settlement + alpha-cut    (qpong.stage4_settlement)

The closed loop is assembled in :mod:`qpong.pipeline`.
"""

__version__ = "1.0.0"

from .config import load_config  # noqa: F401
