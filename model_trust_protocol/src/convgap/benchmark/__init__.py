"""Implied-versus-observed improvement accounting."""

from convgap.benchmark.detectability import (
    Detectability,
    crps_gain_from_sigma_modulation,
    sharpe_gain_from_ic,
)

__all__ = ["Detectability", "crps_gain_from_sigma_modulation", "sharpe_gain_from_ic"]
