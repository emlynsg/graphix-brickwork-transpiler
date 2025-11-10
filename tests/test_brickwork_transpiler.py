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
from graphix.sim.statevec import Statevec
from graphix.simulator import DefaultMeasureMethod
from graphix.transpiler import Circuit
from numpy.random import Generator

from graphix_brickwork_transpiler import nqubits_from_layers, transpile_brickwork, transpile_to_layers

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
    """Test circuit transpilation comparing state vector back-end."""
    pattern = transpile_brickwork(circuit).pattern
    state = circuit.simulate_statevector().statevec
    state_mbqc = pattern.simulate_pattern(rng=fx_rng)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation_og(circuit: Circuit, fx_rng: Generator) -> None:
    """Test circuit transpilation comparing state vector back-end with default transpiler."""
    pattern = transpile_brickwork(circuit).pattern
    pattern.minimize_space()
    pattern_orig = circuit.transpile().pattern
    pattern_orig.minimize_space()
    state_mbqc = pattern.simulate_pattern(rng=fx_rng)
    state_mbqc_orig = pattern_orig.simulate_pattern(rng=fx_rng)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state_mbqc_orig.flatten())) == pytest.approx(1)


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
@pytest.mark.parametrize("check", ["simulation", "flow"])
def test_random_circuit(fx_bg: PCG64, jumps: int, check: str) -> None:
    """Test random circuit transpilation."""
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 6
    circuit = rand_circuit(nqubits, depth, rng, use_ccx=True)
    if check == "simulation":
        test_circuit_simulation(circuit, rng)
    elif check == "flow":
        test_circuit_flow(circuit)


def test_measure(fx_rng: Generator) -> None:
    """Test circuit transpilation with measurement."""
    circuit = Circuit(2)
    circuit.h(1)
    circuit.cnot(0, 1)
    circuit.m(0, Plane.XY, pi / 4)
    transpiled = transpile_brickwork(circuit)
    transpiled.pattern.perform_pauli_measurements()
    transpiled.pattern.minimize_space()

    def simulate_and_measure() -> int:
        measure_method = DefaultMeasureMethod(results=transpiled.pattern.results)
        state = transpiled.pattern.simulate_pattern(rng=fx_rng, measure_method=measure_method)
        measured = measure_method.get_measure_result(transpiled.classical_outputs[0])
        assert isinstance(state, Statevec)
        return measured

    nb_shots = 10000
    count = sum(1 for _ in range(nb_shots) if simulate_and_measure())
    assert abs(count - nb_shots / 2) < nb_shots / 20


class TestBrickworkTranspilerUnitGates:
    """Test the transpiler on circuits with single gates."""

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
    def test_cnot_nonlocal() -> None:
        """Check error for nonlocal CNOTs."""
        circuit = Circuit(3)
        circuit.cnot(0, 2)
        with pytest.raises(ValueError):
            transpile_brickwork(circuit)

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
            circuit.i(0)
        circuit.cnot(0, 1)
        pattern = transpile_brickwork(circuit).pattern
        assert pattern.is_parameterized()
        pattern0 = pattern.subs(alpha, np.pi / 4)
        circuit0 = circuit.subs(alpha, np.pi / 4)
        state = circuit0.simulate_statevector().statevec
        state_mbqc = pattern0.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)
