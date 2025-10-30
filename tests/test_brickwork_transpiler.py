from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from graphix.parameter import Placeholder
from graphix.transpiler import Circuit

from graphix_brickwork_transpiler import nqubits_from_layers, transpile_brickwork, transpile_to_layers  # noqa: PLC2701

if TYPE_CHECKING:
    from numpy.random import Generator


class TestBrickworkTranspilerUnitGates:
    def test_cnot(self, fx_rng: Generator) -> None:
        circuit = Circuit(2)
        circuit.cnot(0, 1)
        pattern = transpile_brickwork(circuit).pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_rx(self, fx_rng: Generator) -> None:  # noqa: D102
        theta = fx_rng.uniform() * 2 * np.pi
        circuit = Circuit(2)
        circuit.rx(0, theta)
        pattern = transpile_brickwork(circuit).pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_rz(self, fx_rng: Generator) -> None:  # noqa: D102
        theta = fx_rng.uniform() * 2 * np.pi
        circuit = Circuit(2)
        circuit.rz(0, theta)
        pattern = transpile_brickwork(circuit).pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_i(self, fx_rng: Generator) -> None:
        circuit = Circuit(2)
        circuit.i(0)
        pattern = transpile_brickwork(circuit).pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_empty_x_z(self, fx_rng: Generator) -> None:
        circuit = Circuit(4)
        for i in [0, 1, 2]:
            for j in [0, 1, 2, 3, 4]:
                circuit.rx(i, j * np.pi / 4)
                circuit.rz(i, j * np.pi / 4)
        pattern = transpile_brickwork(circuit).pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_multi_gate(self, fx_rng: Generator) -> None:
        circuit = Circuit(4)
        circuit.cnot(0, 1)
        circuit.rx(0, np.pi / 4)
        circuit.rz(1, np.pi / 2)
        circuit.cnot(2, 1)
        pattern = transpile_brickwork(circuit).pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_multi_gate_against_default(self, fx_rng: Generator) -> None:
        circuit = Circuit(4)
        circuit.cnot(0, 1)
        circuit.rx(0, np.pi / 4)
        circuit.rz(1, np.pi / 2)
        circuit.cnot(2, 1)
        pattern_default = circuit.transpile().pattern
        state_mbqc_default = pattern_default.simulate_pattern(rng=fx_rng)
        pattern_brickwork = transpile_brickwork(circuit).pattern
        state_mbqc_brickwork = pattern_brickwork.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc_default.flatten().conjugate(), state_mbqc_brickwork.flatten())) == pytest.approx(1)

    def test_multi_gate_deep(self, fx_rng: Generator) -> None:
        circuit = Circuit(4)
        for j in [0, 1, 2, 3, 4]:
            circuit.rx(0, j * np.pi / 4)
            circuit.i(0)
        pattern = transpile_brickwork(circuit).pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_multi_gate_deviant(self, fx_rng: Generator) -> None:
        circuit = Circuit(4)
        circuit.cnot(0, 1)
        circuit.rx(0, np.pi / 4)
        circuit.rz(1, np.pi / 2)
        circuit.cnot(1, 2)
        pattern = transpile_brickwork(circuit, order="deviant").pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_multi_gate_deviant_right(self, fx_rng: Generator) -> None:
        circuit = Circuit(4)
        circuit.cnot(0, 1)
        circuit.rx(0, np.pi / 4)
        circuit.rz(1, np.pi / 2)
        circuit.cnot(1, 2)
        pattern = transpile_brickwork(circuit, order="deviant-right").pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_single_qubit(self, fx_rng: Generator) -> None:
        n = 6
        circuit = Circuit(n)
        for i in range(n - 1):
            circuit.rx(i, 0.0)
            circuit.rz(i, 0.0)
        pattern = transpile_brickwork(circuit).pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_cnot_nonlocal(self) -> None:
        circuit = Circuit(3)
        circuit.cnot(0, 2)
        with pytest.raises(ValueError):
            transpile_brickwork(circuit).pattern

    def test_other_gate(self) -> None:
        circuit = Circuit(1)
        circuit.h(0)
        with pytest.raises(ValueError):
            transpile_brickwork(circuit).pattern

    def test_nqubits_from_layers_0(self) -> None:
        with pytest.raises(ValueError):
            nqubits_from_layers([])

    def test_nqubits_from_layers_n(self) -> None:
        for i in range(2, 4):
            circuit = Circuit(i)
            for j in range(i):
                circuit.rx(j, np.pi / 4)
            layers = transpile_to_layers(circuit)
            assert nqubits_from_layers(layers) == i

    def test_parametrized_circuit(self, fx_rng: Generator) -> None:
        circuit = Circuit(4)
        circuit.cnot(0, 1)
        alpha = Placeholder("alpha")
        for j in [0, 1, 2, 3, 4]:
            circuit.rx(0, j * alpha)
            circuit.i(0)
        pattern = transpile_brickwork(circuit).pattern
        assert pattern.is_parameterized()
        pattern0 = pattern.subs(alpha, np.pi / 4)
        circuit0 = circuit.subs(alpha, np.pi / 4)
        state = circuit0.simulate_statevector().statevec
        state_mbqc = pattern0.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)