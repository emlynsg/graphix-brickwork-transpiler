# Graphix Transpiler from Quantum Circuit to MBQC Patterns via brickwork decomposition

This package provides a transpiler from quantum circuits to MBQC
(Measurement-Based Quantum Computing) patterns via J-∧z decomposition,
designed for use with the [Graphix library](https://github/TeamGraphix/graphix).

In the paper [*Universal Blind Quantum Computation*](https://arxiv.org/abs/0704.1263) by Broadbent, Fitzsimons and Kashefi (2009), circuit-to-graph transpilation leverages the
universality of the Clifford + T gate set ${H, CX, T}$, each of which can be represented by a $2\times 5$ "brick" of nodes. This package
implements that transpilation method to create a "brickwork" open graph using this universal set, and uses the Graphix JCZ transpiler to generate an equivalent MBQC pattern.
