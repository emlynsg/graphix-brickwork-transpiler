"""Graphix Transpiler from circuit to MBQC patterns via brickwork decomposition.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).
"""

from __future__ import annotations

import pytest
from numpy.random import PCG64, Generator

SEED = 25


@pytest.fixture
def fx_bg() -> PCG64:
    """Fixture for bit generator.

    Returns
    -------
        Bit generator

    """
    return PCG64(SEED)


@pytest.fixture
def fx_rng(fx_bg: PCG64) -> Generator:
    """Fixture for generator.

    Returns
    -------
        Generator

    """
    return Generator(fx_bg)
