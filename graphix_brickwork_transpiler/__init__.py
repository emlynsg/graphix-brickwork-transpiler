"""Graphix Transpiler from circuit to MBQC patterns via brickwork decomposition.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).
"""

from __future__ import annotations

from graphix_brickwork_transpiler.brickwork_transpiler import (
    nqubits_from_layers,
    transpile_brickwork,
    transpile_to_layers,
)

__all__ = ["nqubits_from_layers", "transpile_brickwork", "transpile_to_layers"]
