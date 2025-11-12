"""Graphix Transpiler from circuit to MBQC patterns via brickwork decomposition.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).
"""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

import numpy as np
import pytest
from graphix import instruction
from graphix.fundamentals import Plane
from graphix.gflow import find_flow
from graphix.opengraph import OpenGraph
from graphix.parameter import Placeholder
from graphix.random_objects import rand_circuit
from graphix.sim import DensityMatrix
from graphix.transpiler import Circuit
from graphix_jcz_transpiler import (
    CZ,
    J,
    transpile_jcz,
)
from numpy.random import Generator

from graphix_brickwork_transpiler import (
    instruction_to_jcnot,
    nqubits_from_layers,
    transpile_brickwork,
    transpile_to_layers,
)

if TYPE_CHECKING:
    from numpy.random import PCG64

TEST_BASIC_CIRCUITS = [
    Circuit(1, instr=[instruction.H(0)]),
    Circuit(1, instr=[instruction.S(0)]),
    Circuit(1, instr=[instruction.X(0)]),
    Circuit(1, instr=[instruction.Y(0)]),
    Circuit(1, instr=[instruction.Z(0)]),
    Circuit(1, instr=[instruction.I(0)]),
    Circuit(1, instr=[instruction.RX(0, pi / 4)]),
    Circuit(1, instr=[instruction.RY(0, pi / 4)]),
    Circuit(1, instr=[instruction.RZ(0, pi / 4)]),
    Circuit(2, instr=[instruction.CNOT(0, 1)]),
    Circuit(3, instr=[instruction.CCX(0, (1, 2))]),
    Circuit(2, instr=[instruction.RZZ(0, 1, pi / 4)]),
]


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation(circuit: Circuit, fx_rng: Generator) -> None:
    """Test circuit transpilation comparing state vector back-ends."""
    pattern_brickwork = transpile_brickwork(circuit).pattern
    state_original = circuit.simulate_statevector().statevec
    state_brickwork = pattern_brickwork.simulate_pattern(rng=fx_rng)
    dmatrix = DensityMatrix(state_original)
    dmatrix_brickwork = DensityMatrix(data=state_brickwork)  # type: ignore  # noqa: PGH003
    if dmatrix.nqubit != dmatrix_brickwork.nqubit:
        assert dmatrix.nqubit == dmatrix_brickwork.nqubit - 1
        dmatrix_brickwork.ptrace(dmatrix_brickwork.nqubit - 1)
    assert np.abs(np.dot(dmatrix_brickwork.flatten().conjugate(), dmatrix.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation_minimized(circuit: Circuit, fx_rng: Generator) -> None:
    """Test circuit transpilation comparing state vector back-end with default transpiler."""
    pattern_brickwork = transpile_brickwork(circuit).pattern
    pattern_brickwork.minimize_space()
    pattern_brickwork.shift_signals()
    pattern_original = transpile_jcz(circuit).pattern
    pattern_original.minimize_space()
    pattern_original.shift_signals()
    state_original = pattern_original.simulate_pattern(rng=fx_rng)
    state_brickwork = pattern_brickwork.simulate_pattern(rng=fx_rng)
    dmatrix = DensityMatrix(state_original)  # type: ignore  # noqa: PGH003
    dmatrix_brickwork = DensityMatrix(data=state_brickwork)  # type: ignore  # noqa: PGH003
    if dmatrix.nqubit != dmatrix_brickwork.nqubit:
        dmatrix_brickwork.ptrace(dmatrix_brickwork.nqubit - 1)
    assert np.abs(np.dot(dmatrix_brickwork.flatten().conjugate(), dmatrix.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_flow(circuit: Circuit) -> None:
    """Test transpiled circuits have flow."""
    pattern = transpile_brickwork(circuit).pattern
    og = OpenGraph.from_pattern(pattern)
    f, _layers = find_flow(
        og.inside, set(og.inputs), set(og.outputs), {node: meas.plane for node, meas in og.measurements.items()}
    )
    assert f is not None


@pytest.mark.parametrize("jumps", range(1, 11))
def test_random_circuit_simulation(fx_bg: PCG64, jumps: int) -> None:
    """Test random circuit transpilation."""
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 6
    circuit = rand_circuit(nqubits, depth, rng, use_ccx=True)
    test_circuit_simulation(circuit, rng)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_random_circuit_flow(fx_bg: PCG64, jumps: int) -> None:
    """Test random circuit transpilation."""
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 3
    depth = 2
    circuit = rand_circuit(nqubits, depth, rng, use_ccx=True)
    test_circuit_flow(circuit)


class TestBrickworkTranspilerUnitGates:
    """Test the transpiler on circuits with single gates."""

    @staticmethod
    def test_measure() -> None:
        """Test circuit transpilation with measurement."""
        circuit = Circuit(1)
        circuit.m(0, Plane.XY, pi / 4)
        with pytest.raises(ValueError):
            transpile_brickwork(circuit)

    @staticmethod
    def test_j() -> None:
        """Test passing J instruction to transpiler."""
        with pytest.raises(ValueError):
            instruction_to_jcnot(J(0, pi / 4))

    @staticmethod
    def test_cz() -> None:
        """Test passing CZ instruction to transpiler."""
        instr_list = instruction_to_jcnot(CZ((0, 1)))
        comparison_list = [
            [J(0, pi / 2), J(0, pi / 2), J(0, pi / 2), J(0, 0)],
            instruction.CNOT(control=1, target=0),
            [J(0, pi / 2), J(0, pi / 2), J(0, pi / 2), J(0, 0)],
        ]
        assert instr_list == comparison_list

    @staticmethod
    def test_empty_x_z(fx_rng: Generator) -> None:
        """Test with empty X and Z gates."""
        circuit = Circuit(4)
        for i in [0, 1, 2]:
            for j in [0, 1, 2, 3, 4]:
                circuit.rx(i, j * np.pi / 4)
                circuit.rz(i, j * np.pi / 4)
        pattern = transpile_brickwork(circuit).pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @staticmethod
    def test_nqubits_from_layers_0() -> None:
        """Check error for empty layers."""
        with pytest.raises(ValueError):
            nqubits_from_layers([])

    @staticmethod
    def test_nqubits_from_layers_n() -> None:
        """Check nqubits_from_layers function behaves as expected."""
        for i in range(2, 4):
            circuit = Circuit(i)
            for j in range(i):
                circuit.rx(j, np.pi / 4)
            layers = transpile_to_layers(circuit)
            assert nqubits_from_layers(layers) == i

    @staticmethod
    def test_parametrized_circuit(fx_rng: Generator) -> None:
        """Test with parametrized circuit."""
        circuit = Circuit(4)
        circuit.cnot(0, 1)
        alpha = Placeholder("alpha")
        for j in [0, 1, 2, 3, 4]:
            circuit.rx(0, j * alpha)
            circuit.rz(1, j * alpha / 7)
        circuit.cnot(0, 1)
        pattern = transpile_brickwork(circuit).pattern
        assert pattern.is_parameterized()
        pattern0 = pattern.subs(alpha, np.pi / 4)
        circuit0 = circuit.subs(alpha, np.pi / 4)
        state = circuit0.simulate_statevector().statevec
        state_mbqc = pattern0.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)
