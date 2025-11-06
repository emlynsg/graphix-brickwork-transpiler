from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from graphix.parameter import Placeholder
from graphix.transpiler import Circuit
from numpy.random import Generator

from graphix_brickwork_transpiler import nqubits_from_layers, transpile_brickwork, transpile_to_layers

# TODO@emlynsg: Add tests for all other gates.

class TestBrickworkTranspilerUnitGates:
    """Test the transpiler on circuits with single gates."""

    @staticmethod
    def test_cnot(fx_rng: Generator) -> None:
        """Test CNOT transpilation."""
        circuit = Circuit(2)
        circuit.cnot(0, 1)
        pattern = transpile_brickwork(circuit).pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @staticmethod
    def test_rx(fx_rng: Generator) -> None:
        """Test RX transpilation."""
        theta = fx_rng.uniform() * 2 * np.pi
        circuit = Circuit(2)
        circuit.rx(0, theta)
        pattern = transpile_brickwork(circuit).pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @staticmethod
    def test_rz(fx_rng: Generator) -> None:
        """Test RZ transpilation."""
        theta = fx_rng.uniform() * 2 * np.pi
        circuit = Circuit(2)
        circuit.rz(0, theta)
        pattern = transpile_brickwork(circuit).pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @staticmethod
    def test_i(fx_rng: Generator) -> None:
        """Test I transpilation."""
        circuit = Circuit(2)
        circuit.i(0)
        pattern = transpile_brickwork(circuit).pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

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
    def test_multi_gate(fx_rng: Generator) -> None:
        """Test with multiple gates."""
        circuit = Circuit(4)
        circuit.cnot(0, 1)
        circuit.rx(0, np.pi / 4)
        circuit.rz(1, np.pi / 2)
        circuit.cnot(2, 1)
        pattern = transpile_brickwork(circuit).pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @staticmethod
    def test_multi_gate_against_default(fx_rng: Generator) -> None:
        """Test with multiple gates and compare to current Graphix transpiler."""
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

    @staticmethod
    def test_multi_gate_deep(fx_rng: Generator) -> None:
        """Test with multiple gates in multiple layers."""
        circuit = Circuit(4)
        for j in [0, 1, 2, 3, 4]:
            circuit.rx(0, j * np.pi / 4)
            circuit.i(0)
        pattern = transpile_brickwork(circuit).pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @staticmethod
    def test_multi_gate_deviant(fx_rng: Generator) -> None:
        """Test with construction folling the deviant pattern."""
        circuit = Circuit(4)
        circuit.cnot(0, 1)
        circuit.rx(0, np.pi / 4)
        circuit.rz(1, np.pi / 2)
        circuit.cnot(1, 2)
        pattern = transpile_brickwork(circuit, order="deviant").pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @staticmethod
    def test_multi_gate_deviant_right(fx_rng: Generator) -> None:
        """Test with construction folling the deviant right pattern."""
        circuit = Circuit(4)
        circuit.cnot(0, 1)
        circuit.rx(0, np.pi / 4)
        circuit.rz(1, np.pi / 2)
        circuit.cnot(1, 2)
        pattern = transpile_brickwork(circuit, order="deviant-right").pattern
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @staticmethod
    def test_single_qubit(fx_rng: Generator) -> None:
        """Test with single qubit gates."""
        n = 6
        circuit = Circuit(n)
        for i in range(n - 1):
            circuit.rx(i, 0.0)
            circuit.rz(i, 0.0)
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
        pattern = transpile_brickwork(circuit).pattern
        assert pattern.is_parameterized()
        pattern0 = pattern.subs(alpha, np.pi / 4)
        circuit0 = circuit.subs(alpha, np.pi / 4)
        state = circuit0.simulate_statevector().statevec
        state_mbqc = pattern0.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)